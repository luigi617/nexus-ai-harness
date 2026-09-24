from __future__ import annotations

import os
from typing import Any, ClassVar

import boto3
from botocore.config import Config
from dotenv import load_dotenv

from core.message import Message
from core.response import Response
from plugins.models.base import BaseModel
from protocols.tool import Tool

DEFAULT_REGION = "us-east-1"


class BedrockModel(BaseModel):
    provider = "bedrock"
    # USD per 1M tokens: (input, output). Models not listed here cost 0.
    pricing: ClassVar[dict[str, tuple[float, float]]] = {
        "amazon.nova-2-lite-v1:0": (0.33, 2.75),
        "amazon.nova-lite-v1:0": (0.06, 0.24),
        "amazon.nova-micro-v1:0": (0.035, 0.14),
        "amazon.nova-pro-v1:0": (0.8, 3.2),
        "anthropic.claude-fable-5": (10.0, 50.0),
        "anthropic.claude-fable-5-1": (10.0, 50.0),
        "anthropic.claude-haiku-4-5-20251001-v1:0": (1.0, 5.0),
        "anthropic.claude-opus-4-1-20250805-v1:0": (15.0, 75.0),
        "anthropic.claude-opus-4-5-20251101-v1:0": (5.0, 25.0),
        "anthropic.claude-opus-4-6-v1": (5.5, 27.5),
        "anthropic.claude-opus-4-7": (5.0, 25.0),
        "anthropic.claude-opus-4-8": (5.0, 25.0),
        "anthropic.claude-opus-5": (5.0, 25.0),
        "anthropic.claude-opus-5-5": (4.0, 20.0),
        "anthropic.claude-sonnet-4-5-20250929-v1:0": (3.0, 15.0),
        "anthropic.claude-sonnet-4-6": (3.3, 16.5),
        "anthropic.claude-sonnet-5": (2.0, 10.0),
        "apac.amazon.nova-lite-v1:0": (0.063, 0.252),
        "apac.amazon.nova-micro-v1:0": (0.037, 0.148),
        "apac.amazon.nova-pro-v1:0": (0.84, 3.36),
        "apac.anthropic.claude-sonnet-4-20250514-v1:0": (3.0, 15.0),
        "au.anthropic.claude-haiku-4-5-20251001-v1:0": (1.1, 5.5),
        "au.anthropic.claude-opus-4-6-v1": (5.5, 27.5),
        "au.anthropic.claude-opus-4-7": (5.5, 27.5),
        "au.anthropic.claude-opus-4-8": (5.5, 27.5),
        "au.anthropic.claude-opus-5": (5.5, 27.5),
        "au.anthropic.claude-opus-5-5": (4.4, 22.0),
        "au.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.3, 16.5),
        "au.anthropic.claude-sonnet-4-6": (3.3, 16.5),
        "au.anthropic.claude-sonnet-5": (2.2, 11.0),
        "ca.amazon.nova-lite-v1:0": (0.064, 0.256),
        "deepseek.r1-v1:0": (1.35, 5.4),
        "deepseek.v3-v1:0": (0.58, 1.68),
        "deepseek.v3.2": (0.62, 1.85),
        "eu.amazon.nova-2-lite-v1:0": (0.374, 3.157),
        "eu.amazon.nova-lite-v1:0": (0.069, 0.276),
        "eu.amazon.nova-micro-v1:0": (0.04, 0.16),
        "eu.amazon.nova-pro-v1:0": (0.92, 3.68),
        "eu.anthropic.claude-fable-5": (11.0, 55.0),
        "eu.anthropic.claude-haiku-4-5-20251001-v1:0": (1.1, 5.5),
        "eu.anthropic.claude-opus-4-5-20251101-v1:0": (5.5, 27.5),
        "eu.anthropic.claude-opus-4-6-v1": (5.5, 27.5),
        "eu.anthropic.claude-opus-4-7": (5.5, 27.5),
        "eu.anthropic.claude-opus-4-8": (5.5, 27.5),
        "eu.anthropic.claude-opus-5": (5.5, 27.5),
        "eu.anthropic.claude-opus-5-5": (4.4, 22.0),
        "eu.anthropic.claude-sonnet-4-20250514-v1:0": (3.0, 15.0),
        "eu.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.3, 16.5),
        "eu.anthropic.claude-sonnet-4-6": (3.3, 16.5),
        "eu.anthropic.claude-sonnet-5": (2.2, 11.0),
        "eu.mistral.pixtral-large-2502-v1:0": (2.0, 6.0),
        "global.amazon.nova-2-lite-v1:0": (0.3, 2.5),
        "global.anthropic.claude-fable-5": (10.0, 50.0),
        "global.anthropic.claude-fable-5-1": (10.0, 50.0),
        "global.anthropic.claude-haiku-4-5-20251001-v1:0": (1.0, 5.0),
        "global.anthropic.claude-opus-4-5-20251101-v1:0": (5.0, 25.0),
        "global.anthropic.claude-opus-4-6-v1": (5.0, 25.0),
        "global.anthropic.claude-opus-4-7": (5.0, 25.0),
        "global.anthropic.claude-opus-4-8": (5.0, 25.0),
        "global.anthropic.claude-opus-5": (5.0, 25.0),
        "global.anthropic.claude-opus-5-5": (4.0, 20.0),
        "global.anthropic.claude-sonnet-4-20250514-v1:0": (3.0, 15.0),
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.0, 15.0),
        "global.anthropic.claude-sonnet-4-6": (3.0, 15.0),
        "global.anthropic.claude-sonnet-5": (2.0, 10.0),
        "global.moonshotai.kimi-k3": (3.0, 15.0),
        "global.openai.gpt-5.6-luna": (0.2, 1.2),
        "global.openai.gpt-5.6-sol": (4.0, 20.0),
        "global.openai.gpt-5.6-terra": (2.0, 12.0),
        "global.openai.gpt-6-astra": (10.0, 50.0),
        "global.openai.gpt-6-luna": (0.1, 0.5),
        "global.openai.gpt-6-sol": (2.0, 10.0),
        "global.xai.grok-4.6": (2.0, 6.0),
        "google.gemma-3-12b-it": (0.09, 0.29),
        "google.gemma-3-27b-it": (0.23, 0.38),
        "google.gemma-4-26b-a4b": (0.13, 0.4),
        "google.gemma-4-31b": (0.14, 0.4),
        "google.gemma-4-e2b": (0.04, 0.08),
        "in.openai.gpt-5.6-luna": (0.22, 1.32),
        "in.openai.gpt-5.6-terra": (2.2, 13.2),
        "jp.amazon.nova-2-lite-v1:0": (0.396, 3.311),
        "jp.anthropic.claude-haiku-4-5-20251001-v1:0": (1.1, 5.5),
        "jp.anthropic.claude-opus-4-7": (5.5, 27.5),
        "jp.anthropic.claude-opus-4-8": (5.5, 27.5),
        "jp.anthropic.claude-opus-5": (5.5, 27.5),
        "jp.anthropic.claude-opus-5-5": (4.4, 22.0),
        "jp.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.3, 16.5),
        "jp.anthropic.claude-sonnet-4-6": (3.3, 16.5),
        "jp.anthropic.claude-sonnet-5": (2.2, 11.0),
        "meta.llama3-1-70b-instruct-v1:0": (0.72, 0.72),
        "meta.llama3-1-8b-instruct-v1:0": (0.22, 0.22),
        "meta.llama3-3-70b-instruct-v1:0": (0.72, 0.72),
        "meta.llama4-maverick-17b-instruct-v1:0": (0.24, 0.97),
        "meta.llama4-scout-17b-instruct-v1:0": (0.17, 0.66),
        "minimax.minimax-m2": (0.3, 1.2),
        "minimax.minimax-m2.1": (0.3, 1.2),
        "minimax.minimax-m2.5": (0.3, 1.2),
        "mistral.devstral-2-123b": (0.4, 2.0),
        "mistral.magistral-small-2509": (0.5, 1.5),
        "mistral.ministral-3-14b-instruct": (0.2, 0.2),
        "mistral.ministral-3-3b-instruct": (0.1, 0.1),
        "mistral.ministral-3-8b-instruct": (0.15, 0.15),
        "mistral.mistral-large-3-675b-instruct": (0.5, 1.5),
        "mistral.pixtral-large-2502-v1:0": (2.0, 6.0),
        "mistral.voxtral-mini-3b-2507": (0.04, 0.04),
        "mistral.voxtral-small-24b-2507": (0.1, 0.3),
        "moonshot.kimi-k2-thinking": (0.6, 2.5),
        "moonshotai.kimi-k2.5": (0.6, 3.0),
        "nvidia.nemotron-nano-12b-v2": (0.2, 0.6),
        "nvidia.nemotron-nano-3-30b": (0.06, 0.24),
        "nvidia.nemotron-nano-9b-v2": (0.06, 0.23),
        "nvidia.nemotron-super-3-120b": (0.15, 0.65),
        "openai.gpt-5.4": (2.75, 16.5),
        "openai.gpt-5.5": (5.5, 33.0),
        "openai.gpt-5.6-luna": (0.22, 1.32),
        "openai.gpt-5.6-sol": (4.4, 22.0),
        "openai.gpt-5.6-terra": (2.2, 13.2),
        "openai.gpt-6-astra": (11.0, 55.0),
        "openai.gpt-6-luna": (0.11, 0.55),
        "openai.gpt-6-sol": (2.2, 11.0),
        "openai.gpt-oss-120b": (0.15, 0.6),
        "openai.gpt-oss-120b-1:0": (0.15, 0.6),
        "openai.gpt-oss-20b": (0.07, 0.3),
        "openai.gpt-oss-20b-1:0": (0.07, 0.3),
        "openai.gpt-oss-safeguard-120b": (0.15, 0.6),
        "openai.gpt-oss-safeguard-20b": (0.07, 0.2),
        "qwen.qwen3-235b-a22b-2507-v1:0": (0.22, 0.88),
        "qwen.qwen3-32b-v1:0": (0.15, 0.6),
        "qwen.qwen3-coder-30b-a3b-v1:0": (0.15, 0.6),
        "qwen.qwen3-coder-480b-a35b-v1:0": (0.45, 1.8),
        "qwen.qwen3-coder-next": (0.5, 1.2),
        "qwen.qwen3-next-80b-a3b": (0.15, 1.2),
        "qwen.qwen3-vl-235b-a22b": (0.53, 2.66),
        "us-gov.openai.gpt-oss-120b-1:0": (0.18, 0.72),
        "us-gov.openai.gpt-oss-20b-1:0": (0.084, 0.36),
        "us.amazon.nova-2-lite-v1:0": (0.33, 2.75),
        "us.amazon.nova-lite-v1:0": (0.06, 0.24),
        "us.amazon.nova-micro-v1:0": (0.035, 0.14),
        "us.amazon.nova-premier-v1:0": (2.5, 12.5),
        "us.amazon.nova-pro-v1:0": (0.8, 3.2),
        "us.anthropic.claude-fable-5": (11.0, 55.0),
        "us.anthropic.claude-fable-5-1": (11.0, 55.0),
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.1, 5.5),
        "us.anthropic.claude-opus-4-1-20250805-v1:0": (15.0, 75.0),
        "us.anthropic.claude-opus-4-5-20251101-v1:0": (5.5, 27.5),
        "us.anthropic.claude-opus-4-6-v1": (5.5, 27.5),
        "us.anthropic.claude-opus-4-7": (5.5, 27.5),
        "us.anthropic.claude-opus-4-8": (5.5, 27.5),
        "us.anthropic.claude-opus-5": (5.5, 27.5),
        "us.anthropic.claude-opus-5-5": (4.4, 22.0),
        "us.anthropic.claude-sonnet-4-20250514-v1:0": (3.0, 15.0),
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.3, 16.5),
        "us.anthropic.claude-sonnet-4-6": (3.3, 16.5),
        "us.anthropic.claude-sonnet-5": (2.2, 11.0),
        "us.deepseek.r1-v1:0": (1.35, 5.4),
        "us.meta.llama3-1-70b-instruct-v1:0": (0.72, 0.72),
        "us.meta.llama3-1-8b-instruct-v1:0": (0.22, 0.22),
        "us.meta.llama3-3-70b-instruct-v1:0": (0.72, 0.72),
        "us.meta.llama4-maverick-17b-instruct-v1:0": (0.24, 0.97),
        "us.meta.llama4-scout-17b-instruct-v1:0": (0.17, 0.66),
        "us.mistral.pixtral-large-2502-v1:0": (2.0, 6.0),
        "us.moonshotai.kimi-k3": (3.3, 16.5),
        "us.openai.gpt-5.6-luna": (0.22, 1.32),
        "us.openai.gpt-5.6-sol": (4.4, 22.0),
        "us.openai.gpt-5.6-terra": (2.2, 13.2),
        "us.openai.gpt-6-astra": (11.0, 55.0),
        "us.openai.gpt-6-luna": (0.11, 0.55),
        "us.openai.gpt-6-sol": (2.2, 11.0),
        "us.writer.palmyra-x4-v1:0": (2.5, 10.0),
        "us.writer.palmyra-x5-v1:0": (0.6, 6.0),
        "us.xai.grok-4.6": (2.2, 6.6),
        "writer.palmyra-x4-v1:0": (2.5, 10.0),
        "writer.palmyra-x5-v1:0": (0.6, 6.0),
        "xai.grok-4.3": (1.25, 2.5),
        "xai.grok-4.6": (2.2, 6.6),
        "zai.glm-4.7": (0.6, 2.2),
        "zai.glm-4.7-flash": (0.07, 0.4),
        "zai.glm-5": (1.0, 3.2),
    }
    descriptions: ClassVar[dict[str, str]] = {
        "amazon.nova-2-lite-v1:0": "Multimodal reasoning model for visual analysis, planning, and tool use",  # noqa: E501
        "amazon.nova-lite-v1:0": "Efficient model for low-latency assistance, extraction, and routine automation",  # noqa: E501
        "amazon.nova-micro-v1:0": "Efficient model for low-latency assistance, extraction, and routine automation",  # noqa: E501
        "amazon.nova-pro-v1:0": "Flagship model for demanding analysis, coding, and production agent workflows",  # noqa: E501
        "anthropic.claude-fable-5": "Claude model for creative writing, analysis, and controlled agent workflows",  # noqa: E501
        "anthropic.claude-fable-5-1": "Claude model for demanding reasoning and long-horizon agentic work",  # noqa: E501
        "anthropic.claude-haiku-4-5-20251001-v1:0": "Fast Claude model for responsive assistance, classification, and lightweight agents",  # noqa: E501
        "anthropic.claude-opus-4-1-20250805-v1:0": "Flagship Claude model for deep reasoning, coding, and long-horizon agents",  # noqa: E501
        "anthropic.claude-opus-4-5-20251101-v1:0": "Flagship Claude model for deep reasoning, coding, and long-horizon agents",  # noqa: E501
        "anthropic.claude-opus-4-6-v1": "High-end Claude for difficult coding, planning, and slower expert reasoning",  # noqa: E501
        "anthropic.claude-opus-4-7": "Stronger Opus tier for advanced software work and high-stakes reasoning",  # noqa: E501
        "anthropic.claude-opus-4-8": "Top Claude Opus tier for the hardest reasoning, coding, and long-horizon agents",  # noqa: E501
        "anthropic.claude-opus-5": "Strongest Claude Opus model for coding, agents, and professional work",  # noqa: E501
        "anthropic.claude-opus-5-5": "Claude model for long-running agentic coding and knowledge work",  # noqa: E501
        "anthropic.claude-sonnet-4-5-20250929-v1:0": "Balanced Claude model for coding, analysis, agent workflows, and cost control",  # noqa: E501
        "anthropic.claude-sonnet-4-6": "Claude workhorse for coding agents, careful analysis, and production cost control",  # noqa: E501
        "anthropic.claude-sonnet-5": "Everyday Claude agent model for coding, planning, browsing, and general work",  # noqa: E501
        "apac.amazon.nova-lite-v1:0": "Efficient model for low-latency assistance, extraction, and routine automation",  # noqa: E501
        "apac.amazon.nova-micro-v1:0": "Efficient model for low-latency assistance, extraction, and routine automation",  # noqa: E501
        "apac.amazon.nova-pro-v1:0": "Flagship model for demanding analysis, coding, and production agent workflows",  # noqa: E501
        "apac.anthropic.claude-sonnet-4-20250514-v1:0": "Balanced Claude model for coding, analysis, agent workflows, and cost control",  # noqa: E501
        "au.anthropic.claude-haiku-4-5-20251001-v1:0": "Fast Claude model for responsive assistance, classification, and lightweight agents",  # noqa: E501
        "au.anthropic.claude-opus-4-6-v1": "High-end Claude for difficult coding, planning, and slower expert reasoning",  # noqa: E501
        "au.anthropic.claude-opus-4-7": "Stronger Opus tier for advanced software work and high-stakes reasoning",  # noqa: E501
        "au.anthropic.claude-opus-4-8": "Top Claude Opus tier for the hardest reasoning, coding, and long-horizon agents",  # noqa: E501
        "au.anthropic.claude-opus-5": "Strongest Claude Opus model for coding, agents, and professional work",  # noqa: E501
        "au.anthropic.claude-opus-5-5": "Claude model for long-running agentic coding and knowledge work",  # noqa: E501
        "au.anthropic.claude-sonnet-4-5-20250929-v1:0": "Balanced Claude model for coding, analysis, agent workflows, and cost control",  # noqa: E501
        "au.anthropic.claude-sonnet-4-6": "Claude workhorse for coding agents, careful analysis, and production cost control",  # noqa: E501
        "au.anthropic.claude-sonnet-5": "Everyday Claude agent model for coding, planning, browsing, and general work",  # noqa: E501
        "ca.amazon.nova-lite-v1:0": "Efficient model for low-latency assistance, extraction, and routine automation",  # noqa: E501
        "deepseek.r1-v1:0": "Classic open reasoning model for transparent math, coding, and deliberate problem solving",  # noqa: E501
        "deepseek.v3-v1:0": "Hybrid-reasoning DeepSeek model with thinking and non-thinking modes",  # noqa: E501
        "deepseek.v3.2": "Hybrid-reasoning DeepSeek model with thinking and non-thinking modes, sparse attention, and tool-use",  # noqa: E501
        "eu.amazon.nova-2-lite-v1:0": "Multimodal reasoning model for visual analysis, planning, and tool use",  # noqa: E501
        "eu.amazon.nova-lite-v1:0": "Efficient model for low-latency assistance, extraction, and routine automation",  # noqa: E501
        "eu.amazon.nova-micro-v1:0": "Efficient model for low-latency assistance, extraction, and routine automation",  # noqa: E501
        "eu.amazon.nova-pro-v1:0": "Flagship model for demanding analysis, coding, and production agent workflows",  # noqa: E501
        "eu.anthropic.claude-fable-5": "Claude model for creative writing, analysis, and controlled agent workflows",  # noqa: E501
        "eu.anthropic.claude-haiku-4-5-20251001-v1:0": "Fast Claude model for responsive assistance, classification, and lightweight agents",  # noqa: E501
        "eu.anthropic.claude-opus-4-5-20251101-v1:0": "Flagship Claude model for deep reasoning, coding, and long-horizon agents",  # noqa: E501
        "eu.anthropic.claude-opus-4-6-v1": "High-end Claude for difficult coding, planning, and slower expert reasoning",  # noqa: E501
        "eu.anthropic.claude-opus-4-7": "Stronger Opus tier for advanced software work and high-stakes reasoning",  # noqa: E501
        "eu.anthropic.claude-opus-4-8": "Top Claude Opus tier for the hardest reasoning, coding, and long-horizon agents",  # noqa: E501
        "eu.anthropic.claude-opus-5": "Strongest Claude Opus model for coding, agents, and professional work",  # noqa: E501
        "eu.anthropic.claude-opus-5-5": "Claude model for long-running agentic coding and knowledge work",  # noqa: E501
        "eu.anthropic.claude-sonnet-4-20250514-v1:0": "Balanced Claude model for coding, analysis, agent workflows, and cost control",  # noqa: E501
        "eu.anthropic.claude-sonnet-4-5-20250929-v1:0": "Balanced Claude model for coding, analysis, agent workflows, and cost control",  # noqa: E501
        "eu.anthropic.claude-sonnet-4-6": "Claude workhorse for coding agents, careful analysis, and production cost control",  # noqa: E501
        "eu.anthropic.claude-sonnet-5": "Everyday Claude agent model for coding, planning, browsing, and general work",  # noqa: E501
        "eu.mistral.pixtral-large-2502-v1:0": "Mistral vision-language model for image understanding and multimodal chat",  # noqa: E501
        "global.amazon.nova-2-lite-v1:0": "Multimodal reasoning model for visual analysis, planning, and tool use",  # noqa: E501
        "global.anthropic.claude-fable-5": "Claude model for creative writing, analysis, and controlled agent workflows",  # noqa: E501
        "global.anthropic.claude-fable-5-1": "Claude model for demanding reasoning and long-horizon agentic work",  # noqa: E501
        "global.anthropic.claude-haiku-4-5-20251001-v1:0": "Fast Claude model for responsive assistance, classification, and lightweight agents",  # noqa: E501
        "global.anthropic.claude-opus-4-5-20251101-v1:0": "Flagship Claude model for deep reasoning, coding, and long-horizon agents",  # noqa: E501
        "global.anthropic.claude-opus-4-6-v1": "High-end Claude for difficult coding, planning, and slower expert reasoning",  # noqa: E501
        "global.anthropic.claude-opus-4-7": "Stronger Opus tier for advanced software work and high-stakes reasoning",  # noqa: E501
        "global.anthropic.claude-opus-4-8": "Top Claude Opus tier for the hardest reasoning, coding, and long-horizon agents",  # noqa: E501
        "global.anthropic.claude-opus-5": "Strongest Claude Opus model for coding, agents, and professional work",  # noqa: E501
        "global.anthropic.claude-opus-5-5": "Claude model for long-running agentic coding and knowledge work",  # noqa: E501
        "global.anthropic.claude-sonnet-4-20250514-v1:0": "Balanced Claude model for coding, analysis, agent workflows, and cost control",  # noqa: E501
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0": "Balanced Claude model for coding, analysis, agent workflows, and cost control",  # noqa: E501
        "global.anthropic.claude-sonnet-4-6": "Claude workhorse for coding agents, careful analysis, and production cost control",  # noqa: E501
        "global.anthropic.claude-sonnet-5": "Everyday Claude agent model for coding, planning, browsing, and general work",  # noqa: E501
        "global.moonshotai.kimi-k3": "Multimodal Kimi model with 1M context and toggleable max-effort thinking for long-horizon agent work",  # noqa: E501
        "global.openai.gpt-5.6-luna": "Cost-efficient GPT-5.6 model for fast, high-volume workloads",  # noqa: E501
        "global.openai.gpt-5.6-sol": "Frontier GPT-5.6 model for complex professional work, coding, and agentic workflows",  # noqa: E501
        "global.openai.gpt-5.6-terra": "Balanced GPT-5.6 model for capable, cost-efficient everyday work",  # noqa: E501
        "global.openai.gpt-6-astra": "GPT-6 Astra is OpenAI's most capable model for complex reasoning, coding, computer use, research, and document creation.",  # noqa: E501
        "global.openai.gpt-6-luna": "OpenAI's most efficient model for focused, high-volume tasks",  # noqa: E501
        "global.openai.gpt-6-sol": "OpenAI model for complex coding and agentic workflows",  # noqa: E501
        "global.xai.grok-4.6": "xAI's frontier model for long-running agents, coding, knowledge work, and visual projects",  # noqa: E501
        "google.gemma-3-12b-it": "Open multimodal Gemma instruction model for multilingual text generation and image understanding",  # noqa: E501
        "google.gemma-3-27b-it": "Largest open Gemma 3 instruction model for multilingual text generation and visual understanding",  # noqa: E501
        "google.gemma-4-26b-a4b": "Open Gemma instruction model for efficient chat and self-hosted deployments",  # noqa: E501
        "google.gemma-4-31b": "Largest Gemma 4 instruction model for open, self-hosted chat and reasoning",  # noqa: E501
        "google.gemma-4-e2b": "Open Gemma instruction model for efficient chat and self-hosted deployments",  # noqa: E501
        "in.openai.gpt-5.6-luna": "Cost-efficient GPT-5.6 model for fast, high-volume workloads",  # noqa: E501
        "in.openai.gpt-5.6-terra": "Balanced GPT-5.6 model for capable, cost-efficient everyday work",  # noqa: E501
        "jp.amazon.nova-2-lite-v1:0": "Multimodal reasoning model for visual analysis, planning, and tool use",  # noqa: E501
        "jp.anthropic.claude-haiku-4-5-20251001-v1:0": "Fast Claude model for responsive assistance, classification, and lightweight agents",  # noqa: E501
        "jp.anthropic.claude-opus-4-7": "Stronger Opus tier for advanced software work and high-stakes reasoning",  # noqa: E501
        "jp.anthropic.claude-opus-4-8": "Top Claude Opus tier for the hardest reasoning, coding, and long-horizon agents",  # noqa: E501
        "jp.anthropic.claude-opus-5": "Strongest Claude Opus model for coding, agents, and professional work",  # noqa: E501
        "jp.anthropic.claude-opus-5-5": "Claude model for long-running agentic coding and knowledge work",  # noqa: E501
        "jp.anthropic.claude-sonnet-4-5-20250929-v1:0": "Balanced Claude model for coding, analysis, agent workflows, and cost control",  # noqa: E501
        "jp.anthropic.claude-sonnet-4-6": "Claude workhorse for coding agents, careful analysis, and production cost control",  # noqa: E501
        "jp.anthropic.claude-sonnet-5": "Everyday Claude agent model for coding, planning, browsing, and general work",  # noqa: E501
        "meta.llama3-1-70b-instruct-v1:0": "Open Llama instruction model for multilingual chat, reasoning, and coding",  # noqa: E501
        "meta.llama3-1-8b-instruct-v1:0": "Compact open Llama model for lightweight chat, drafting, and self-hosting",  # noqa: E501
        "meta.llama3-3-70b-instruct-v1:0": "Popular open Llama workhorse for multilingual chat, coding, and self-hosting",  # noqa: E501
        "meta.llama4-maverick-17b-instruct-v1:0": "Open multimodal Llama for strong reasoning with efficient everyday serving",  # noqa: E501
        "meta.llama4-scout-17b-instruct-v1:0": "Open Llama with long-context vision for efficient multimodal agents",  # noqa: E501
        "minimax.minimax-m2": "Efficient open MiniMax model built for coding agents and tool-heavy workflows",  # noqa: E501
        "minimax.minimax-m2.1": "Earlier MiniMax agent model for practical coding and productivity tasks",  # noqa: E501
        "minimax.minimax-m2.5": "Prior MiniMax coding model for agent workflows, office edits, and automation",  # noqa: E501
        "mistral.devstral-2-123b": "Mistral's coding-agent model for repository work, terminal tasks, and software fixes",  # noqa: E501
        "mistral.magistral-small-2509": "Open multimodal reasoning model for transparent analysis of text and images",  # noqa: E501
        "mistral.ministral-3-14b-instruct": "Open vision-language model for efficient local deployment, instruction following, and tool use",  # noqa: E501
        "mistral.ministral-3-3b-instruct": "Compact open vision-language model for edge deployment, instruction following, and tool use",  # noqa: E501
        "mistral.ministral-3-8b-instruct": "Compact open vision-language model for edge deployment, instruction following, and tool use",  # noqa: E501
        "mistral.mistral-large-3-675b-instruct": "Mistral's largest general model for enterprise agents, coding, and multilingual reasoning",  # noqa: E501
        "mistral.pixtral-large-2502-v1:0": "Mistral vision-language model for image understanding and multimodal chat",  # noqa: E501
        "mistral.voxtral-mini-3b-2507": "Open audio-language model for speech transcription, audio understanding, and voice-driven tool use",  # noqa: E501
        "mistral.voxtral-small-24b-2507": "Open audio-language model for speech transcription, audio understanding, and voice-driven tool use",  # noqa: E501
        "moonshot.kimi-k2-thinking": "Thinking Kimi model for slower research passes, planning, and hard technical questions",  # noqa: E501
        "moonshotai.kimi-k2.5": "Earlier Kimi frontier model for long-context agents, coding, and multimodal work",  # noqa: E501
        "nvidia.nemotron-nano-12b-v2": "Nemotron multimodal model for visual reasoning and agentic AI workflows",  # noqa: E501
        "nvidia.nemotron-nano-3-30b": "Small Nemotron 3 MoE for efficient coding, math, and long-context agents",  # noqa: E501
        "nvidia.nemotron-nano-9b-v2": "Compact Nemotron model for efficient reasoning and deployable AI agents",  # noqa: E501
        "nvidia.nemotron-super-3-120b": "Nemotron middle tier for collaborative agents and high-volume reasoning workloads",  # noqa: E501
        "openai.gpt-5.4": "Agent-ready GPT for coding and computer-use workflows at a lower cost",  # noqa: E501
        "openai.gpt-5.5": "Default frontier GPT for coding, computer use, research, and knowledge work",  # noqa: E501
        "openai.gpt-5.6-luna": "Cost-efficient GPT-5.6 model for fast, high-volume workloads",  # noqa: E501
        "openai.gpt-5.6-sol": "Frontier GPT-5.6 model for complex professional work, coding, and agentic workflows",  # noqa: E501
        "openai.gpt-5.6-terra": "Balanced GPT-5.6 model for capable, cost-efficient everyday work",  # noqa: E501
        "openai.gpt-6-astra": "GPT-6 Astra is OpenAI's most capable model for complex reasoning, coding, computer use, research, and document creation.",  # noqa: E501
        "openai.gpt-6-luna": "OpenAI's most efficient model for focused, high-volume tasks",  # noqa: E501
        "openai.gpt-6-sol": "OpenAI model for complex coding and agentic workflows",
        "openai.gpt-oss-120b": "Open GPT reasoning model for self-hosted agents and controllable deployments",  # noqa: E501
        "openai.gpt-oss-120b-1:0": "Open GPT reasoning model for self-hosted agents and controllable deployments",  # noqa: E501
        "openai.gpt-oss-20b": "Open GPT reasoning model for self-hosted agents and controllable deployments",  # noqa: E501
        "openai.gpt-oss-20b-1:0": "Open GPT reasoning model for self-hosted agents and controllable deployments",  # noqa: E501
        "openai.gpt-oss-safeguard-120b": "Safety model for policy screening, moderation, and risk-aware routing workflows",  # noqa: E501
        "openai.gpt-oss-safeguard-20b": "Safety model for policy screening, moderation, and risk-aware routing workflows",  # noqa: E501
        "qwen.qwen3-235b-a22b-2507-v1:0": "Updated large open Qwen3 MoE instruct model for multilingual chat, coding, and tool use",  # noqa: E501
        "qwen.qwen3-32b-v1:0": "Dense open Qwen model for self-hosted chat, reasoning, and coding",  # noqa: E501
        "qwen.qwen3-coder-30b-a3b-v1:0": "Smaller Qwen coder for efficient local agents and repo-level fixes",  # noqa: E501
        "qwen.qwen3-coder-480b-a35b-v1:0": "Open Qwen coding heavyweight for repository reasoning and agentic engineering",  # noqa: E501
        "qwen.qwen3-coder-next": "Open-weight Qwen coding model for agents, repository edits, and multi-turn tool use",  # noqa: E501
        "qwen.qwen3-next-80b-a3b": "Qwen instruction model for multilingual chat, reasoning, and tool use",  # noqa: E501
        "qwen.qwen3-vl-235b-a22b": "Qwen vision-language instruct model for visual reasoning, documents, and agent tasks",  # noqa: E501
        "us-gov.openai.gpt-oss-120b-1:0": "Open GPT reasoning model for self-hosted agents and controllable deployments",  # noqa: E501
        "us-gov.openai.gpt-oss-20b-1:0": "Open GPT reasoning model for self-hosted agents and controllable deployments",  # noqa: E501
        "us.amazon.nova-2-lite-v1:0": "Multimodal reasoning model for visual analysis, planning, and tool use",  # noqa: E501
        "us.amazon.nova-lite-v1:0": "Efficient model for low-latency assistance, extraction, and routine automation",  # noqa: E501
        "us.amazon.nova-micro-v1:0": "Efficient model for low-latency assistance, extraction, and routine automation",  # noqa: E501
        "us.amazon.nova-premier-v1:0": "Multimodal model for complex analysis, long-context understanding, tool use, and model distillation",  # noqa: E501
        "us.amazon.nova-pro-v1:0": "Flagship model for demanding analysis, coding, and production agent workflows",  # noqa: E501
        "us.anthropic.claude-fable-5": "Claude model for creative writing, analysis, a",
        "us.anthropic.claude-fable-5-1": "Claude model for demanding reasoning and lon",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": "Fast Claude model for responsi",
        "us.anthropic.claude-opus-4-1-20250805-v1:0": "Flagship Claude model for deep",
        "us.anthropic.claude-opus-4-5-20251101-v1:0": "Flagship Claude model for deep",
        "us.anthropic.claude-opus-4-6-v1": "High-end Claude for difficult coding, plan",
        "us.anthropic.claude-opus-4-7": "Stronger Opus tier for advanced software work",
        "us.anthropic.claude-opus-4-8": "most capable; hard reasoning, complex tasks",
        "us.anthropic.claude-opus-5": "Strongest Claude Opus model for coding, agents,",
        "us.anthropic.claude-opus-5-5": "Claude model for long-running agentic coding",
        "us.anthropic.claude-sonnet-4-20250514-v1:0": "Balanced Claude model for codin",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "Balanced Claude model for cod",
        "us.anthropic.claude-sonnet-4-6": "Claude workhorse for coding agents, careful",
        "us.anthropic.claude-sonnet-5": "Everyday Claude agent model for coding, plann",
        "us.deepseek.r1-v1:0": "Classic open reasoning model for transparent math, coding, and deliberate problem solving",  # noqa: E501
        "us.meta.llama3-1-70b-instruct-v1:0": "Open Llama instruction model for multilingual chat, reasoning, and coding",  # noqa: E501
        "us.meta.llama3-1-8b-instruct-v1:0": "Compact open Llama model for lightweight chat, drafting, and self-hosting",  # noqa: E501
        "us.meta.llama3-3-70b-instruct-v1:0": "Popular open Llama workhorse for multilingual chat, coding, and self-hosting",  # noqa: E501
        "us.meta.llama4-maverick-17b-instruct-v1:0": "Open multimodal Llama for strong reasoning with efficient everyday serving",  # noqa: E501
        "us.meta.llama4-scout-17b-instruct-v1:0": "Open Llama with long-context vision for efficient multimodal agents",  # noqa: E501
        "us.mistral.pixtral-large-2502-v1:0": "Mistral vision-language model for image understanding and multimodal chat",  # noqa: E501
        "us.moonshotai.kimi-k3": "Multimodal Kimi model with 1M context and toggleable max-effort thinking for long-horizon agent work",  # noqa: E501
        "us.openai.gpt-5.6-luna": "Cost-efficient GPT-5.6 model for fast, high-volume workloads",  # noqa: E501
        "us.openai.gpt-5.6-sol": "Frontier GPT-5.6 model for complex professional work, coding, and agentic workflows",  # noqa: E501
        "us.openai.gpt-5.6-terra": "Balanced GPT-5.6 model for capable, cost-efficient everyday work",  # noqa: E501
        "us.openai.gpt-6-astra": "GPT-6 Astra is OpenAI's most capable model for complex reasoning, coding, computer use, research, and document creation.",  # noqa: E501
        "us.openai.gpt-6-luna": "OpenAI's most efficient model for focused, high-volume tasks",  # noqa: E501
        "us.openai.gpt-6-sol": "OpenAI model for complex coding and agentic workflows",
        "us.writer.palmyra-x4-v1:0": "Enterprise language model for workflow automation, coding, data analysis, and tool use",  # noqa: E501
        "us.writer.palmyra-x5-v1:0": "Reasoning model for deliberate analysis, multi-step problem solving, and tool use",  # noqa: E501
        "us.xai.grok-4.6": "xAI's frontier model for long-running agents, coding, knowledge work, and visual projects",  # noqa: E501
        "writer.palmyra-x4-v1:0": "Enterprise language model for workflow automation, coding, data analysis, and tool use",  # noqa: E501
        "writer.palmyra-x5-v1:0": "Reasoning model for deliberate analysis, multi-step problem solving, and tool use",  # noqa: E501
        "xai.grok-4.3": "xAI's default Grok for chat, coding, agentic tools, and lower hallucination risk",  # noqa: E501
        "xai.grok-4.6": "xAI's frontier model for long-running agents, coding, knowledge work, and visual projects",  # noqa: E501
        "zai.glm-4.7": "Mature GLM model for dependable coding, reasoning, and structured agent tasks",  # noqa: E501
        "zai.glm-4.7-flash": "Budget GLM lane for fast coding help, routing, and everyday automation",  # noqa: E501
        "zai.glm-5": "General GLM flagship for coding, analysis, and tool-heavy engineering workflows",  # noqa: E501
    }

    def __init__(
        self,
        model: str,
        region: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        **params,
    ) -> None:
        self.name = model
        self.description = self.descriptions.get(model, "")
        self.params = params
        load_dotenv()
        self.region = region or os.getenv("AWS_REGION") or DEFAULT_REGION
        self.api_key = api_key or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        self.max_tokens = max_tokens
        # Accepted explicitly so it is not swept into self.params, where it would
        # pollute inferenceConfig (Converse rejects unknown members).
        self.timeout = timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if self.api_key:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.api_key
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(connect_timeout=self.timeout, read_timeout=self.timeout),
            )
        return self._client

    def _generate(self, history: list[Message], tools: list[Tool]) -> Response:
        system, messages = self._to_converse(history)
        kwargs: dict[str, Any] = {
            "modelId": self.name,
            "messages": messages,
            "inferenceConfig": {"maxTokens": self.max_tokens, **self.params},
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["toolConfig"] = {"tools": [self._tool_spec(t) for t in tools]}
        response = self._get_client().converse(**kwargs)
        parsed = self._parse(response)
        parsed.cost = self._cost(parsed.usage)
        return parsed

    @staticmethod
    def _to_converse(history: list[Message]) -> tuple[list[dict], list[dict]]:
        system: list[dict] = []
        messages: list[dict] = []
        pending_results: list[dict] = []

        def add_turn(role: str, blocks: list[dict]) -> None:
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": role, "content": list(blocks)})

        def flush_results() -> None:
            if pending_results:
                add_turn("user", pending_results)
                pending_results.clear()

        for m in history:
            if m.role == "tool":
                # Batch consecutive tool results into one user turn.
                pending_results.append(
                    {
                        "toolResult": {
                            "toolUseId": m.tool_use_id,
                            "content": [{"text": m.content}],
                        }
                    }
                )
                continue

            flush_results()
            if m.role == "system":
                system.append({"text": m.content})
            elif m.role == "user":
                add_turn("user", [{"text": m.content}])
            elif m.role == "assistant":
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"text": m.content})
                for call in m.tool_calls:
                    blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": call.get("id"),
                                "name": call.get("name"),
                                "input": call.get("arguments", {}),
                            }
                        }
                    )
                if blocks:  # an empty assistant turn is rejected by the API
                    add_turn("assistant", blocks)

        flush_results()
        return system, messages

    @staticmethod
    def _tool_spec(tool: Tool) -> dict:
        return {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "json": tool.parameters or {"type": "object", "properties": {}}
                },
            }
        }

    @staticmethod
    def _parse(response: dict) -> Response:
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                use = block["toolUse"]
                tool_calls.append(
                    {
                        "id": use.get("toolUseId"),
                        "name": use.get("name"),
                        "arguments": use.get("input", {}),
                    }
                )
        usage = response.get("usage", {})
        return Response(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage={
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
            },
        )
