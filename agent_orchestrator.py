"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore
"""

import boto3
import json
import time
import os
import sys
import uuid
import random
import logging
import re
import io
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Tuple

# Ensure the parent directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Strands Agents SDK
from strands import Agent
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key, Attr

import config
from agent_observability import (          # pre-written: @tool + X-Ray + CloudWatch
    tool, apply_observability_config, tracer, setup_logging, flush_logs, print_trace_hint,
)
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------
# OUTPUT UTILITIES
# -----------------------------------------------------
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)


# -----------------------------------------------------
# AWS CLIENTS & DYNAMODB TABLES
# -----------------------------------------------------
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)

# DynamoDB Table References
orders_table   = dynamodb.Table(config.ORDERS_TABLE)
customers_table = dynamodb.Table(config.CUSTOMERS_TABLE)
workflow_table  = dynamodb.Table(config.WORKFLOW_STATE_TABLE)


# -------------------------------------------------------
# WORKFLOW STATE - HELPER FUNCTIONS
# -------------------------------------------------------

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """Create a blank WorkflowState record at the start of a new customer session."""
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """Read the current WorkflowState for a session."""
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """Update WorkflowState with optimistic locking."""
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


def _apply_guardrail(**model_kwargs) -> BedrockModel:
    """Build a BedrockModel and attach the guardrail named in GUARDRAIL_ID / GUARDRAIL_VERSION."""
    guardrail_id = os.environ.get("GUARDRAIL_ID") or getattr(config, "GUARDRAIL_ID", "")
    guardrail_version = os.environ.get("GUARDRAIL_VERSION") or getattr(config, "GUARDRAIL_VERSION", "")
    if guardrail_id and str(guardrail_version).isdigit():
        model_kwargs.update(
            guardrail_id=guardrail_id,
            guardrail_version=str(guardrail_version),
            guardrail_trace="enabled",
        )
    return BedrockModel(**model_kwargs)


# -------------------------------------------------------
# TASK 2 - MULTI-AGENT ORCHESTRATION
# -------------------------------------------------------

# -------------------------------------------------------
# 2.A - INVENTORY AGENT
# -------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """Convert DynamoDB types (Decimal etc.) into plain JSON-safe Python values."""
    return json.loads(json.dumps(obj, default=str))


def _find_order(order_id: str, customer_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Locate an order by order_id, using a direct get_item when customer_id is known."""
    if customer_id:
        res = orders_table.get_item(
            Key={"customer_id": customer_id, "order_id": order_id}
        )
        if "Item" in res:
            return res["Item"]
    res = orders_table.scan(FilterExpression=Attr("order_id").eq(order_id))
    items = res.get("Items", [])
    return items[0] if items else None


@tool
def check_order_status(order_id: str, customer_id: str = None) -> Dict[str, Any]:
    """Look up the details and current status of a single order in DynamoDB.

    Args:
        order_id: The order identifier, for example "ORD-27176".
        customer_id: Optional customer identifier (for example "CUST-001").
            When supplied, a direct key lookup is used; otherwise the order
            is located by order_id alone.

    Returns:
        A dictionary with the order record (status, items, purchase date, etc.),
        or {"error": "..."} if the order does not exist.
    """
    item = _find_order(order_id, customer_id)
    return _jsonable(item) if item else {"error": f"Order {order_id} not found"}


@tool
def get_customer_tier(customer_id: str) -> str:
    """Fetch the loyalty tier of a customer from DynamoDB.

    Args:
        customer_id: The customer identifier, for example "CUST-001".

    Returns:
        The tier as a string, either "Standard" or "Premium". Defaults to
        "Standard" when the customer record has no tier.
    """
    response = customers_table.get_item(Key={"customer_id": customer_id})
    item = response.get("Item", {})
    return item.get("tier", "Standard")


@tool
def list_customer_orders(customer_id: str) -> List[Dict[str, Any]]:
    """List every order that belongs to a customer.

    Args:
        customer_id: The customer identifier, for example "CUST-001".

    Returns:
        A list of order records (possibly empty) for that customer.
    """
    response = orders_table.query(
        IndexName="CustomerIndex",
        KeyConditionExpression="customer_id = :cid",
        ExpressionAttributeValues={":cid": customer_id}
    )
    return _jsonable(response.get("Items", []))


def build_inventory_agent() -> Agent:
    """Construct the InventoryAgent (WORKER_MODEL_ID, temperature=0.1)."""
    model = _apply_guardrail(model_id=config.WORKER_MODEL_ID, temperature=0.1)
    system_prompt = (
        "You are the Inventory Agent. Your sole responsibility is to gather order facts, "
        "order status, and customer tier information from DynamoDB. "
        "Do not make return or refund decisions."
    )
    return Agent(
        name="InventoryAgent",
        model=model,
        system_prompt=system_prompt,
        tools=[check_order_status, get_customer_tier, list_customer_orders]
    )


# -------------------------------------------------------
# 2.B - REFUND AGENT
# -------------------------------------------------------

@tool
def get_inventory_context(session_id: str) -> dict:
    """Retrieve the order and customer context gathered so far for this session.

    Args:
        session_id: The workflow session identifier.

    Returns:
        The current WorkflowState record (including any inventory_agent and
        policy_agent results already stored), or an empty dict if none exists.
    """
    state = _read_workflow_state(session_id)
    return _jsonable(state) if state else {}


@tool
def initiate_refund(session_id: str, order_id: str, amount: float) -> Dict[str, Any]:
    """Execute a refund: mark the order REFUNDED and record it in the workflow state.

    Args:
        session_id: The workflow session identifier.
        order_id: The order to refund, for example "ORD-27176".
        amount: The refund amount in dollars.

    Returns:
        {"status": "SUCCESS", "order_id": ..., "amount": ...} on success, or
        {"status": "FAILED", "error": "..."} if the order cannot be found.
    """
    order = _find_order(order_id)
    if not order:
        return {"status": "FAILED", "error": f"Order {order_id} not found"}

    orders_table.update_item(
        Key={"customer_id": order["customer_id"], "order_id": order_id},
        UpdateExpression="SET #s = :status",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":status": "REFUNDED"}
    )
    workflow_table.update_item(
        Key={"session_id": session_id},
        UpdateExpression="SET refund_status = :rs",
        ExpressionAttributeValues={":rs": f"SUCCESS - ${amount:.2f}"}
    )
    return {"status": "SUCCESS", "order_id": order_id, "amount": amount}


def build_refund_agent() -> Agent:
    """Construct the RefundAgent (WORKER_MODEL_ID, temperature=0.1)."""
    model = _apply_guardrail(model_id=config.WORKER_MODEL_ID, temperature=0.1)
    system_prompt = (
        "You are a Refund Specialist Agent.\n"
        "Policy Rules:\n"
        "- Standard Customers: Maximum return window is 30 days.\n"
        "- Premium Customers: Maximum return window is 60 days.\n"
        "Verify customer tier and check if the purchase date is within the allowed "
        "window before initiating refunds."
    )
    return Agent(
        name="RefundAgent",
        model=model,
        system_prompt=system_prompt,
        tools=[get_inventory_context, initiate_refund]
    )


# -------------------------------------------------------
# 2.C - POLICY AGENT
# -------------------------------------------------------

def build_policy_agent() -> Agent:
    """Construct the PolicyAgent: three KB retriever sub-agents queried in parallel."""
    retriever_model = _apply_guardrail(model_id=config.WORKER_MODEL_ID, temperature=0.0)

    @tool
    def retrieve_returns(query: str) -> str:
        """Search the Returns Knowledge Base.

        Args:
            query: The customer's question about returns.

        Returns:
            Matching return-policy passages as text.
        """
        return format_kb_results(retrieve_from_knowledge_base(config.RETURNS_KB_ID, query))

    @tool
    def retrieve_shipping(query: str) -> str:
        """Search the Shipping Knowledge Base.

        Args:
            query: The customer's question about shipping.

        Returns:
            Matching shipping-policy passages as text.
        """
        return format_kb_results(retrieve_from_knowledge_base(config.SHIPPING_KB_ID, query))

    @tool
    def retrieve_warranty(query: str) -> str:
        """Search the Warranty Knowledge Base.

        Args:
            query: The customer's question about warranty coverage.

        Returns:
            Matching warranty-policy passages as text.
        """
        return format_kb_results(retrieve_from_knowledge_base(config.WARRANTY_KB_ID, query))

    returns_retriever = Agent(
        name="ReturnsPolicyRetrieverAgent",
        model=retriever_model,
        system_prompt="Call your retrieval tool with the user's query and return the passages found.",
        tools=[retrieve_returns]
    )
    shipping_retriever = Agent(
        name="ShippingPolicyRetrieverAgent",
        model=retriever_model,
        system_prompt="Call your retrieval tool with the user's query and return the passages found.",
        tools=[retrieve_shipping]
    )
    warranty_retriever = Agent(
        name="WarrantyPolicyRetrieverAgent",
        model=retriever_model,
        system_prompt="Call your retrieval tool with the user's query and return the passages found.",
        tools=[retrieve_warranty]
    )

    @tool
    def search_all_policies(query: str) -> dict:
        """Search the Returns, Shipping and Warranty knowledge bases in parallel.

        Args:
            query: The customer's policy question.

        Returns:
            A dict with keys "returns", "shipping" and "warranty", each holding
            the text returned by the matching retriever agent.
        """
        retrievers = {
            "returns": returns_retriever,
            "shipping": shipping_retriever,
            "warranty": warranty_retriever,
        }
        results = {}
        trace.kb_start({
            "Returns": config.RETURNS_KB_ID,
            "Shipping": config.SHIPPING_KB_ID,
            "Warranty": config.WARRANTY_KB_ID,
        })
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_key = {
                executor.submit(agent, query): key
                for key, agent in retrievers.items()
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = str(future.result())
                except Exception as e:
                    results[key] = f"ERROR retrieving {key} policy: {e}"
        trace.kb_done(len(retrievers))
        for key in ("returns", "shipping", "warranty"):
            trace.kb_result(key.capitalize(), results[key])
        return results

    coordinator_model = _apply_guardrail(model_id=config.WORKER_MODEL_ID, temperature=0.2)
    return Agent(
        name="PolicyAgent",
        model=coordinator_model,
        system_prompt=(
            "You are the Policy Agent. For every policy question call search_all_policies "
            "once, then summarise the returns, shipping and warranty results accurately."
        ),
        tools=[search_all_policies]
    )


# -------------------------------------------------------
#  2.D - COMMUNICATION AGENT
# -------------------------------------------------------

@tool
def get_full_workflow_context(session_id: str) -> Dict[str, Any]:
    """Read the complete workflow record for a session.

    Args:
        session_id: The workflow session identifier.

    Returns:
        The full WorkflowState record (inventory, policy and refund results),
        or an empty dict if the session does not exist.
    """
    response = workflow_table.get_item(Key={"session_id": session_id})
    return _jsonable(response.get("Item", {}))


def build_communication_agent() -> Agent:
    """Construct the CommunicationAgent (WORKER_MODEL_ID, temperature=0.3)."""
    model = _apply_guardrail(model_id=config.WORKER_MODEL_ID, temperature=0.3)
    system_prompt = (
        "You are the Communication Agent. Synthesize findings from inventory, policy, "
        "and refund operations into a polite, empathetic, and professional response for the customer."
    )
    return Agent(
        name="CommunicationAgent",
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context]
    )


# -------------------------------------------------------
# 2.E - ORCHESTRATOR AGENT
# -------------------------------------------------------

def _state_context(state: dict) -> str:
    """Render the current WorkflowState as text to prepend to a worker prompt."""
    return json.dumps(_jsonable(state), indent=2)


def build_orchestrator_agent(
    inventory_agent: Agent,
    refund_agent: Agent,
    policy_agent: Agent,
    communication_agent: Agent
) -> Agent:

    @tool
    def initialize_session(session_id: str, customer_id: str) -> Dict[str, Any]:
        """Create the WorkflowState record for a new session. Call this first.

        Args:
            session_id: The session id from the message prefix.
            customer_id: The customer id from the message prefix.

        Returns:
            The newly created WorkflowState record.
        """
        return _jsonable(_create_workflow_state(session_id, customer_id))

    @tool
    def route_to_inventory_agent(session_id: str, prompt: str) -> str:
        """Send an order, order-history, customer-tier or account question to InventoryAgent.

        Args:
            session_id: The workflow session identifier.
            prompt: The task for InventoryAgent.

        Returns:
            InventoryAgent's findings as text (also saved as inventory_agent).
        """
        state = _read_workflow_state(session_id)              # 1. read state
        if state is None:
            return f"ERROR: no WorkflowState for session {session_id}; call initialize_session first."
        expected_version = int(state["version"])
        trace.step_start("inventory_agent")
        trace.agent_section("InventoryAgent")
        response = str(inventory_agent(                       # 2. call worker with context
            f"[Session ID: {session_id}]\n[Workflow state]\n{_state_context(state)}\n\n{prompt}"
        ))
        _update_workflow_state(                               # 3. persist with version check
            session_id, {"inventory_agent": response}, expected_version
        )
        trace.step_done("inventory_agent", expected_version)
        return response

    @tool
    def route_to_policy_agent(session_id: str, prompt: str) -> str:
        """Send a returns, shipping or warranty POLICY question to PolicyAgent.

        Args:
            session_id: The workflow session identifier.
            prompt: The policy question for PolicyAgent.

        Returns:
            PolicyAgent's answer as text (also saved as policy_agent).
        """
        state = _read_workflow_state(session_id)
        if state is None:
            return f"ERROR: no WorkflowState for session {session_id}; call initialize_session first."
        expected_version = int(state["version"])
        trace.step_start("policy_agent")
        trace.agent_section("PolicyAgent")
        response = str(policy_agent(
            f"[Session ID: {session_id}]\n[Workflow state]\n{_state_context(state)}\n\n{prompt}"
        ))
        _update_workflow_state(
            session_id, {"policy_agent": response}, expected_version
        )
        trace.step_done("policy_agent", expected_version)
        return response

    @tool
    def route_to_refund_agent(session_id: str, prompt: str) -> str:
        """Send a return or refund PROCESSING request to RefundAgent.

        Args:
            session_id: The workflow session identifier.
            prompt: The refund task, including the order id and amount.

        Returns:
            RefundAgent's outcome as text (also saved as refund_agent).
        """
        state = _read_workflow_state(session_id)
        if state is None:
            return f"ERROR: no WorkflowState for session {session_id}; call initialize_session first."
        expected_version = int(state["version"])
        trace.step_start("refund_agent")
        trace.agent_section("RefundAgent")
        response = str(refund_agent(
            f"[Session ID: {session_id}]\n[Workflow state]\n{_state_context(state)}\n\n{prompt}"
        ))
        _update_workflow_state(
            session_id, {"refund_agent": response}, expected_version
        )
        trace.step_done("refund_agent", expected_version)
        return response

    @tool
    def route_to_communication_agent(session_id: str, prompt: str) -> str:
        """Hand off to CommunicationAgent to write the customer reply. Always the LAST call.

        Args:
            session_id: The workflow session identifier.
            prompt: Instruction to compose the final customer response.

        Returns:
            The customer-facing response text (also saved as communication_agent).
        """
        state = _read_workflow_state(session_id)
        if state is None:
            return f"ERROR: no WorkflowState for session {session_id}; call initialize_session first."
        expected_version = int(state["version"])
        trace.step_start("communication_agent")
        trace.agent_section("CommunicationAgent")
        response = str(communication_agent(
            f"[Session ID: {session_id}]\n[Workflow state]\n{_state_context(state)}\n\n{prompt}"
        ))
        _update_workflow_state(
            session_id, {"communication_agent": response}, expected_version
        )
        trace.step_done("communication_agent", expected_version)
        return response

    model = _apply_guardrail(model_id=config.ORCHESTRATOR_MODEL_ID, temperature=0.0)

    system_prompt = (
        "You are the Master Orchestrator Agent. You coordinate specialist agents and "
        "NEVER write the final customer-facing response yourself.\n\n"
        "Every user message starts with [Session ID: ...] and [Customer ID: ...]. "
        "Use those values in every tool call.\n\n"
        "Routing rules:\n"
        "1. Every request, always: call initialize_session FIRST.\n"
        "2. Order status, return or refund requests: route_to_inventory_agent first, "
        "then route_to_refund_agent.\n"
        "3. Policy meaning questions (return windows, shipping rates, warranty terms): "
        "route_to_policy_agent.\n"
        "4. Account questions (\"what is my tier?\", \"am I premium?\"): "
        "route_to_inventory_agent. NEVER route account questions to the policy agent - "
        "it only knows policy text, not customer data.\n"
        "5. Math or calculation questions: go straight to route_to_communication_agent. "
        "Skip the Inventory, Policy and Refund agents.\n"
        "6. Every request, always (last step): route_to_communication_agent composes the final reply. "
        "It MUST be your very last tool call - no exceptions, and no tool may follow it."
    )

    return Agent(
        name="OrchestratorAgent",
        model=model,
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent
        ]
    )

# -------------------------------------------------------
# TASK 3 - GUARDRAILS & DEPLOYMENT
# -------------------------------------------------------

def create_guardrail() -> Tuple[str, str]:
    """Create (or update) the Bedrock Guardrail and publish a numbered version."""
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)
    guardrail_name = config.GUARDRAIL_NAME

    guardrail_kwargs = dict(
        name=guardrail_name,
        description="Guardrail for NovaMart Multi-Agent System",
        crossRegionConfig={"guardrailProfileIdentifier": "us.guardrail.v1:0"},
        contentPolicyConfig={
            'filtersConfig': [
                {'type': 'SEXUAL', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'VIOLENCE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'HATE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'INSULTS', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
                {'type': 'MISCONDUCT', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'}
            ]
        },
        topicPolicyConfig={
            'tierConfig': {'tierName': 'STANDARD'},
            'topicsConfig': [
                {
                    'name': 'CompetitorProducts',
                    'definition': 'Inquiries regarding rival products or external retail competitors.',
                    'examples': ['Do you sell items from Amazon?', 'Is Walmart better than NovaMart?'],
                    'type': 'DENY'
                },
                {
                    'name': 'PricingNegotiations',
                    'definition': 'Haggling, or requests to change an advertised price. Calculations that use a specified price and discount are allowed.',
                    'examples': ['Can you give me a discount?', 'Lower the price for me.', 'Will you match a cheaper price if I ask?'],
                    'type': 'DENY'
                },
                {
                    'name': 'LegalThreats',
                    'definition': 'Statements threatening legal action or lawsuits against NovaMart.',
                    'examples': ['I will sue your company.', 'Contacting my lawyer.'],
                    'type': 'DENY'
                }
            ]
        },
        sensitiveInformationPolicyConfig={
            'piiEntitiesConfig': [
                {'type': 'CREDIT_DEBIT_CARD_NUMBER', 'action': 'BLOCK'},
                {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'BLOCK'},
                {'type': 'EMAIL', 'action': 'ANONYMIZE'},
                {'type': 'PHONE', 'action': 'ANONYMIZE'}
            ]
        },
        wordPolicyConfig={
            'managedWordListsConfig': [
                {'type': 'PROFANITY'}
            ]
        },
        blockedInputMessaging="I cannot process this request due to content policy restrictions.",
        blockedOutputsMessaging="I cannot fulfill this request as the generated response violates policy guidelines."
    )

    def _find_guardrail_id():
        paginator = bedrock_client.get_paginator('list_guardrails')
        for page in paginator.paginate():
            for g in page.get('guardrails', []):
                if g.get('name') == guardrail_name:
                    return g['id']
        return None

    # Step 1: create; if one with this name already exists, update it so the
    # policies always match the specification (e.g. STANDARD tier, new topic text).
    try:
        guardrail_id = bedrock_client.create_guardrail(**guardrail_kwargs)['guardrailId']
        print(f"  Created guardrail {guardrail_id}")
    except bedrock_client.exceptions.ConflictException:
        guardrail_id = _find_guardrail_id()
        if not guardrail_id:
            raise RuntimeError(f"Guardrail '{guardrail_name}' exists but could not be found")
        bedrock_client.update_guardrail(guardrailIdentifier=guardrail_id, **guardrail_kwargs)
        print(f"  Updated existing guardrail {guardrail_id}")

    # Step 2: publish a numbered version (never DRAFT)
    version_response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="Release version"
    )
    guardrail_version = str(version_response['version'])
    print(f"  Published guardrail version {guardrail_version}")
    return guardrail_id, guardrail_version


def _upsert_env_var(key: str, value: str) -> None:
    """Write KEY=value into the project's .env file (create or replace the line)."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    lines = []
    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = [l.rstrip("\n") for l in f if not l.startswith(f"{key}=")]
    lines.append(f"{key}={value}")
    with open(env_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def deploy_to_agentcore_runtime(orchestrator, guardrail_id: str, guardrail_version: str) -> str:
    """Configure and deploy the runtime with the supplied agentcore_cli helper."""
    import agentcore_cli

    if not str(guardrail_version).isdigit():
        raise ValueError(f"GUARDRAIL_VERSION must be a numbered version, got {guardrail_version!r}")

    env_vars = {
        "AWS_REGION": config.AWS_REGION,
        "PROJECT_NAME": config.PROJECT_NAME,
        "RETURNS_KB_ID": config.RETURNS_KB_ID,
        "SHIPPING_KB_ID": config.SHIPPING_KB_ID,
        "WARRANTY_KB_ID": config.WARRANTY_KB_ID,
        "AGENT_LOG_GROUP": config.AGENT_LOG_GROUP,
        "GUARDRAIL_ID": guardrail_id,
        "GUARDRAIL_VERSION": str(guardrail_version),
    }

    agentcore_cli.stage_runtime_code()             # build/runtime/ = what the CLI packages
    agentcore_cli.configure_runtime(
        env_vars,
        network_mode="PUBLIC",
        protocol="HTTP",
        execution_role_arn=config.AGENTCORE_ROLE_ARN,
    )
    agentcore_cli.deploy()
    runtime_arn = agentcore_cli.deployed_runtime_arn()
    if not runtime_arn:
        raise RuntimeError("AgentCore deploy finished but no runtime ARN was returned")

    _upsert_env_var("AGENTCORE_RUNTIME_ARN", runtime_arn)
    return runtime_arn


# -------------------------------------------------------
# TASK 4 - MEMORY
# -------------------------------------------------------

def configure_memory(runtime_arn: str) -> str:
    """Create (or reuse) an AgentCore Memory with a session-summary strategy, 7-day retention."""
    memory_name = config.MEMORY_NAME   # letters, digits and underscores only

    memory_id = None
    for mem in agentcore_control.list_memories().get("memories", []):
        if str(mem.get("id", "")).startswith(memory_name):
            memory_id = mem["id"]
            break

    if memory_id is None:
        resp = agentcore_control.create_memory(
            name=memory_name,
            description="Session memory for NovaMart multi-agent system",
            eventExpiryDuration=7,
            memoryStrategies=[{
                "summaryMemoryStrategy": {
                    "name": "SessionSummarizer",
                    "namespaces": ["/summaries/{actorId}/{sessionId}"],
                }
            }],
        )
        memory_id = resp["memory"]["id"]

    # Wait for the memory resource to become ACTIVE
    deadline = time.time() + 600
    while time.time() < deadline:
        mem = agentcore_control.get_memory(memoryId=memory_id)["memory"]
        if mem["status"] == "ACTIVE":
            return mem["arn"]
        if mem["status"] == "FAILED":
            raise RuntimeError(f"Memory {memory_id} failed: {mem.get('failureReason')}")
        time.sleep(10)
    raise TimeoutError(f"Memory {memory_id} not ACTIVE after 600s")


# -------------------------------------------------------
# TASK 6 - OBSERVABILITY
# -------------------------------------------------------

def configure_observability(runtime_arn: str) -> Dict[str, Any]:
    """Enable CloudWatch Logs + X-Ray tracing for the runtime via apply_observability_config()."""
    logging_configuration = {
        "cloudWatchConfig": {
            "logGroupName": config.AGENT_LOG_GROUP,
            "logLevel": "INFO",
            "enabled": True,
        },
        "xRayConfig": {
            "enabled": True,
            "samplingRate": 1.0,
        },
    }
    return apply_observability_config(runtime_arn, logging_configuration)


# -------------------------------------------------------
# AGENTCORE GATEWAY DEPLOYMENT
# -------------------------------------------------------

_ORDERS_FUNCTION    = os.environ.get('ORDERS_FUNCTION',    f"{config.PROJECT_NAME}-orders-api")
_POLICY_FUNCTION    = os.environ.get('POLICY_FUNCTION',    f"{config.PROJECT_NAME}-policy-api")
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', f"{config.PROJECT_NAME}-customers-api")


def _gw_get_function_arn(function_name: str) -> str:
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} -> target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    print(f"\n  Registering {len(targets)} Gateway targets...")
    for t in targets:
        try:
            lambda_arn = _gw_get_function_arn(t['function'])
            _gw_create_target(agentcore_ctrl, gateway_id, t, lambda_arn)
        except Exception as e:
            print(f"    [Skipped] {t['name']}: {e}")

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url, 'status': 'CREATING'}


# -------------------------------------------------------
# RUNTIME INVOCATION
# -------------------------------------------------------

# -------------------------------------------------------
# DEPLOYMENT ENTRY POINT
# -------------------------------------------------------

def deploy_all():
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        print(f"  Gateway URL : {gw['gateway_url']}")
        print(f"  Agents connect via MCP at this endpoint — no code changes needed")
    except Exception as e:
        print(f"  [Note] Gateway deployment skipped: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    return runtime_arn, guardrail_id


# -------------------------------------------------------
# AGENTCORE RUNTIME - SERVE MODE
# -------------------------------------------------------

_RUNTIME_MARKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agentcore-runtime")


def serve_runtime() -> None:
    """Run as the AgentCore Runtime HTTP app (started with no arguments inside build/runtime/)."""
    from bedrock_agentcore.runtime import BedrockAgentCoreApp

    os.environ["AGENT_RUNTIME_MODE"] = "1"
    setup_logging()
    app = BedrockAgentCoreApp()
    orchestrator = build_orchestrator_agent(
        build_inventory_agent(), build_refund_agent(),
        build_policy_agent(), build_communication_agent()
    )

    @app.entrypoint
    def invoke(payload, context=None):
        """Payload: {"prompt": ..., "session_id": ..., "customer_id": ...}."""
        prompt = (payload or {}).get("prompt", "")
        if not prompt:
            return "Please provide a customer support question."
        session_id = payload.get("session_id") or str(uuid.uuid4())[:8]
        customer_id = payload.get("customer_id", "anonymous_customer")
        full_prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {prompt}"
        try:
            with tracer.trace_request(session_id, customer_id, prompt):
                return str(orchestrator(full_prompt))
        finally:
            flush_logs()

    app.run()


if __name__ == '__main__':
    if len(sys.argv) == 1 and os.path.exists(_RUNTIME_MARKER):
        serve_runtime()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == 'deploy':
        deploy_all()

    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("Running local agent test...")
        inventory_agent     = build_inventory_agent()
        refund_agent        = build_refund_agent()
        policy_agent        = build_policy_agent()
        communication_agent = build_communication_agent()
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )

        test_cases = [
            ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
            ("CUST-002", "What is the return policy for premium customers?"),
            ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
        ]
        for customer_id, query in test_cases:
            session_id = str(uuid.uuid4())[:8]
            print(f"\n{'-'*60}")
            print(f"Session: {session_id} | Customer: {customer_id}")
            print(f"Query: {query}")
            prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
            with tracer.trace_request(session_id, customer_id, query):
                response = orchestrator(prompt)
            print(f"Response: {response}")
            print_trace_hint()

    elif len(sys.argv) > 1 and sys.argv[1] == 'chat':
        W = _C.W

        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

        print()
        print(f"  {_C.GRY}{'-' * W}{_C.RESET}")
        print(f"  {_C.BOLD}Test Customers{_C.RESET}")
        print(f"  {_C.GRY}{'-' * W}{_C.RESET}")
        print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
        print(f"  {_C.GRY}{'-'*8}  {'-'*16}  {'-'*8}  {'-'*10}  {'-'*20}{_C.RESET}")
        for cid, name, tier, order, product in [
            ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Sony headphones"),
            ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "mechanical keyboard"),
            ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "laptop"),
            ("CUST-004", "David Lee",     "Standard", "ORD-30001", "phone case"),
        ]:
            tier_col = _C.INV if tier == 'Premium' else _C.GRY
            print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
                  f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
        print(f"  {_C.GRY}{'-' * W}{_C.RESET}")
        print()

        customer_id = (
            input(f"  Enter Customer ID (default: CUST-001): ").strip()
            or "CUST-001"
        )
        session_id  = str(uuid.uuid4())[:8]
        print()
        print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
        print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
        print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
        print()

        print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
        inventory_agent     = build_inventory_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  InventoryAgent{_C.RESET}",    flush=True)
        refund_agent        = build_refund_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  RefundAgent{_C.RESET}",       flush=True)
        policy_agent        = build_policy_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  PolicyAgent{_C.RESET}",       flush=True)
        communication_agent = build_communication_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  CommunicationAgent{_C.RESET}", flush=True)
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  Orchestrator{_C.RESET}",      flush=True)
        print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
        print()

        while True:
            try:
                user_input = input(
                    f"  {_C.BOLD}You >{_C.RESET} "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
                break

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print(f"  {_C.GRY}Session ended.{_C.RESET}")
                break

            prompt  = (f"[Session ID: {session_id}] "
                       f"[Customer ID: {customer_id}] {user_input}")
            t0_turn = time.time()

            trace.new_turn()
            sys.stdout = _trace_writer
            try:
                with tracer.trace_request(session_id, customer_id, user_input):
                    response = orchestrator(prompt)
            finally:
                sys.stdout = _real_stdout

            elapsed = time.time() - t0_turn

            final_state = _read_workflow_state(session_id) or {}
            comm_result = final_state.get('communication_agent', '')
            text = _strip_xml_tags(comm_result or str(response))

            trace.summary(session_id, elapsed)

            print()
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            for line in text.splitlines():
                print(f"  {line}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print()

    else:
        print("Usage:")
        print("  python agent_orchestrator.py deploy  # Deploy to AgentCore")
        print("  python agent_orchestrator.py test    # Run automated test cases")
        print("  python agent_orchestrator.py chat    # Interactive terminal chat")