"""
OpenRouter client for AI-powered trading decisions.
Interfaces with Grok models through OpenRouter for market analysis and trading strategies.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime

import httpx
from openai import AsyncOpenAI
from json_repair import repair_json

from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin, log_error_with_context
from src.utils.prompts import MULTI_AGENT_PROMPT_TPL


@dataclass
class TradingDecision:
    """Represents an AI trading decision."""
    action: str  # "buy", "sell", "hold"
    side: str    # "yes", "no"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    position_size_pct: float  # Percentage of available capital to use
    max_price: Optional[float] = None  # Maximum price willing to pay
    stop_loss: Optional[float] = None  # Stop loss price
    take_profit: Optional[float] = None  # Take profit price
    expected_return: Optional[float] = None  # Expected return percentage
    risk_assessment: Optional[str] = None  # Risk level assessment


@dataclass
class MarketAnalysis:
    """Represents AI market analysis."""
    market_id: str
    prediction: str
    probability_estimate: float  # 0.0 to 1.0
    key_factors: List[str]
    risks: List[str]
    opportunities: List[str]
    time_horizon: str
    analysis_quality: float  # 0.0 to 1.0 - how good the analysis is
    cost: float  # API cost for this analysis


class OpenAIClient(TradingLoggerMixin):
    """
    OpenAI client for AI-powered trading decisions.
    Uses OpenAI models for market analysis and trading strategy.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (defaults to settings)
        """
        self.api_key = api_key or settings.api.openai_api_key
        self.base_url = settings.api.openai_base_url
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=3
        )

        # Model configuration
        self.primary_model = settings.trading.primary_model
        self.fallback_model = settings.trading.fallback_model
        self.temperature = settings.trading.ai_temperature
        self.max_tokens = settings.trading.ai_max_tokens
        
        # Cost tracking
        self.total_cost = 0.0
        self.request_count = 0
        
        self.logger.info(
            "OpenAI client initialized",
            primary_model=self.primary_model,
            fallback_model=self.fallback_model
        )
    
    async def get_trading_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str
    ) -> TradingDecision:
        """
        Get a trading decision from the AI model.
        """
        prompt = self._prepare_prompt(market_data, portfolio_data, news_summary)
        messages = [{"role": "user", "content": prompt}]

        response_content, cost = await self._make_completion_request(messages)

        # The model is expected to return a dialogue, with the Trader's response being a JSON object.
        # We need to extract that JSON object from the response.
        try:
            # Find the last JSON object in the response
            json_str = response_content[response_content.rfind('{'):response_content.rfind('}')+1]
            decision_json = self._parse_json_response(json_str, "trading_decision")
        except (ValueError, IndexError):
            self.logger.error("Failed to extract or parse JSON from model response.", response=response_content)
            raise ValueError("Invalid JSON response from model")

        return TradingDecision(
            action=decision_json.get("action", "SKIP"),
            side=decision_json.get("side"),
            confidence=decision_json.get("confidence", 0.0),
            reasoning=decision_json.get("rationale", "No rationale provided."),
            position_size_pct=0 # This can be enhanced later
        )

    def _prepare_prompt(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str
    ) -> str:
        """
        Prepare the prompt for the AI model.
        """
        days_to_expiry = (datetime.fromtimestamp(market_data['expiration_ts']) - datetime.now()).days
        
        # Calculate dynamic max trade value based on portfolio balance
        available_balance = portfolio_data.get("available_balance", 0)
        max_trade_value = available_balance * (settings.trading.max_position_size_pct / 100)
        
        prompt_params = {
            "title": market_data['title'],
            "yes_price": (market_data.get('yes_bid', 0) + market_data.get('yes_ask', 100)) / 2,
            "no_price": (market_data.get('no_bid', 0) + market_data.get('no_ask', 100)) / 2,
            "volume": market_data['volume'],
            "days_to_expiry": days_to_expiry,
            "news_summary": news_summary,
            "cash": available_balance,
            "volume_min": settings.trading.min_volume,
            "max_days_to_expiry": settings.trading.max_time_to_expiry_days,
            "ev_threshold": settings.trading.min_confidence_to_trade * 100,
            "max_trade_value": max_trade_value,
            "max_position_pct": settings.trading.max_position_size_pct
        }
        return MULTI_AGENT_PROMPT_TPL.format(**prompt_params)

    async def _make_completion_request(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Tuple[str, float]:
        """
        Make a completion request with cost tracking and aggressive fallback logic.
        
        Args:
            messages: Chat messages
            model: Model to use
            temperature: Response randomness
            max_tokens: Maximum tokens in response
            response_format: Response format specification
            max_retries: Maximum number of retries for failed requests
        
        Returns:
            Tuple of (response_content, cost)
        """
        model = model or self.primary_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        last_exception = None
        original_model = model  # Keep track of original model
        fallback_used = False
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = await self.client.chat.completions.create(**kwargs)
                
                processing_time = time.time() - start_time
                
                # Check if response is valid
                if not response.choices or not response.choices[0].message.content:
                    if attempt < max_retries:
                        self.logger.warning(
                            f"Empty response on attempt {attempt + 1}, retrying...",
                            model=model,
                            attempt=attempt + 1,
                            max_retries=max_retries
                        )
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        raise ValueError("Received empty response from OpenAI API")
                
                response_content = response.choices[0].message.content
                
                # Log the actual response for debugging
                self.logger.info(
                    f"Received response from {model}",
                    response_length=len(response_content) if response_content else 0,
                    response_preview=response_content[:100] if response_content else "EMPTY",
                    response_format_required=response_format is not None
                )
                
                # Estimate cost (rough approximation)
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                
                # Rough cost estimation for OpenAI models (adjust as needed)
                cost = (input_tokens * 0.00001) + (output_tokens * 0.00002)
                
                self.total_cost += cost
                self.request_count += 1
                
                self.logger.debug(
                    "AI completion request successful",
                    model=model,
                    original_model=original_model,
                    fallback_used=fallback_used,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                    processing_time=processing_time,
                    attempt=attempt + 1
                )
                
                return response_content, cost
                
            except Exception as e:
                last_exception = e
                
                # Enhanced fallback logic: Try fallback if primary model fails and we haven't used fallback yet
                if (not fallback_used and 
                    original_model == self.primary_model and 
                    self.fallback_model != self.primary_model):
                    
                    self.logger.warning(
                        "Primary model failed, switching to fallback for remaining attempts",
                        primary_model=self.primary_model,
                        fallback_model=self.fallback_model,
                        attempt=attempt + 1,
                        error=str(e)
                    )
                    model = self.fallback_model
                    fallback_used = True
                
                if attempt < max_retries:
                    self.logger.warning(
                        f"Request failed on attempt {attempt + 1}, retrying...",
                        model=model,
                        original_model=original_model,
                        fallback_used=fallback_used,
                        error=str(e),
                        attempt=attempt + 1,
                        max_retries=max_retries
                    )
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    break
        
        # If all retries failed with primary model, try fallback one more time
        if (not fallback_used and 
            original_model == self.primary_model and 
            self.fallback_model != self.primary_model):
            
            self.logger.warning(
                "All attempts with primary model failed, making final attempt with fallback",
                primary_model=self.primary_model,
                fallback_model=self.fallback_model
            )
            
            try:
                kwargs = {
                    "model": self.fallback_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = await self.client.chat.completions.create(**kwargs)
                
                if response.choices and response.choices[0].message.content:
                    response_content = response.choices[0].message.content
                    
                    # Estimate cost
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    cost = (input_tokens * 0.00001) + (output_tokens * 0.00002)
                    
                    self.total_cost += cost
                    self.request_count += 1
                    
                    self.logger.info(
                        "Fallback model succeeded after primary model failure",
                        primary_model=self.primary_model,
                        fallback_model=self.fallback_model,
                        cost=cost
                    )
                    
                    return response_content, cost
                    
            except Exception as fallback_error:
                self.logger.error(
                    "Both primary and fallback models failed",
                    primary_model=self.primary_model,
                    fallback_model=self.fallback_model,
                    primary_error=str(last_exception),
                    fallback_error=str(fallback_error)
                )
        
        # All retries failed
        log_error_with_context(
            last_exception,
            {
                "model": original_model,
                "final_model_attempted": model,
                "fallback_used": fallback_used,
                "messages_count": len(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "attempts": max_retries + 1
            },
            "openrouter_completion_all_retries_failed"
        )
        raise last_exception

    def _parse_json_response(self, response_content: str, context: str) -> Dict[str, Any]:
        """Parses a JSON response, with a retry mechanism to repair it."""
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            self.logger.warning("Initial JSON parsing failed, attempting to repair.", response=response_content)
            repaired_json_str = self._repair_json_response(response_content)
            if repaired_json_str:
                try:
                    return json.loads(repaired_json_str)
                except json.JSONDecodeError:
                    self.logger.error("JSON parsing failed even after repair.", repaired_response=repaired_json_str)
            
            raise ValueError("Failed to parse JSON response")

    def _repair_json_response(self, response_content: str) -> Optional[str]:
        """Repairs a malformed JSON string."""
        try:
            return repair_json(response_content)
        except Exception as e:
            self.logger.error("JSON repair failed.", error=str(e))
            return None

    def _get_fallback_response(self, context: str) -> Dict[str, Any]:
        """Returns a fallback response when JSON parsing fails."""
        self.logger.error("JSON parsing and repair failed, returning fallback.", context=context)
        raise ValueError("Failed to get a valid JSON response from the model.")

    async def close(self) -> None:
        """Close the OpenAI client session."""
        if self.client:
            await self.client.close()
            self.logger.info(
                "OpenAI client closed",
                total_cost=self.total_cost,
                total_requests=self.request_count
            ) 