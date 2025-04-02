import re
from typing import List, Dict, Any
import pandas as pd
import logging


class FinancialAnalyzer:
    """
    Specialized financial text analyzer that extracts metrics and calculates ratios.
    Focuses on extracting structured financial data from unstructured text.
    """
    
    def __init__(self):
        """
        Initialize the financial analyzer with logging.
        """
        self.logger = logging.getLogger(__name__)
    
    def extract_metrics(self, text: str) -> Dict[str, Any]:
        """
        Extracts key financial metrics from text using regex patterns.
        Designed to handle common financial notation patterns.
        
        Args:
            text: Text containing financial information
            
        Returns:
            Dictionary of extracted financial metrics
        """
        metrics = {}
        
        try:
            # Extract revenue
            revenue_pattern = r"(?:Revenue|Sales).*?[$€£](\d+(?:\.\d+)?)\s*(million|billion|m|b)"
            revenue = re.search(revenue_pattern, text, re.IGNORECASE)
            if revenue:
                metrics["revenue"] = self._normalize_value(revenue.group(1), revenue.group(2))
            
            # Extract profit/net income
            profit_pattern = r"(?:Net Income|Profit|Earnings).*?[$€£](\d+(?:\.\d+)?)\s*(million|billion|m|b)"
            profit = re.search(profit_pattern, text, re.IGNORECASE)
            if profit:
                metrics["net_income"] = self._normalize_value(profit.group(1), profit.group(2))
            
            # Extract EPS
            eps_pattern = r"(?:EPS|Earnings Per Share).*?[$€£](\d+(?:\.\d+)?)"
            eps = re.search(eps_pattern, text, re.IGNORECASE)
            if eps:
                metrics["eps"] = float(eps.group(1))
            
            # Extract operating expenses
            opex_pattern = r"(?:Operating Expenses|OPEX).*?[$€£](\d+(?:\.\d+)?)\s*(million|billion|m|b)"
            opex = re.search(opex_pattern, text, re.IGNORECASE)
            if opex:
                metrics["operating_expenses"] = self._normalize_value(opex.group(1), opex.group(2))
            
            # Extract cost of goods sold
            cogs_pattern = r"(?:Cost of Goods Sold|COGS).*?[$€£](\d+(?:\.\d+)?)\s*(million|billion|m|b)"
            cogs = re.search(cogs_pattern, text, re.IGNORECASE)
            if cogs:
                metrics["cost_of_goods"] = self._normalize_value(cogs.group(1), cogs.group(2))
            
            # Extract total assets
            assets_pattern = r"(?:Total Assets|Assets).*?[$€£](\d+(?:\.\d+)?)\s*(million|billion|m|b)"
            assets = re.search(assets_pattern, text, re.IGNORECASE)
            if assets:
                metrics["total_assets"] = self._normalize_value(assets.group(1), assets.group(2))
            
            # Extract total liabilities
            liabilities_pattern = r"(?:Total Liabilities|Liabilities).*?[$€£](\d+(?:\.\d+)?)\s*(million|billion|m|b)"
            liabilities = re.search(liabilities_pattern, text, re.IGNORECASE)
            if liabilities:
                metrics["total_liabilities"] = self._normalize_value(liabilities.group(1), liabilities.group(2))
            
            # Extract market cap
            market_cap_pattern = r"(?:Market Cap|Market Capitalization).*?[$€£](\d+(?:\.\d+)?)\s*(million|billion|m|b)"
            market_cap = re.search(market_cap_pattern, text, re.IGNORECASE)
            if market_cap:
                metrics["market_cap"] = self._normalize_value(market_cap.group(1), market_cap.group(2))
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error extracting metrics: {str(e)}")
            return metrics
    
    def _normalize_value(self, value: str, unit: str) -> float:
        """
        Normalizes financial values to a standard format.
        Converts abbreviations to actual values.
        
        Args:
            value: The numerical value as a string
            unit: The unit (million, billion, etc.)
            
        Returns:
            Normalized value as a float
        """
        value = float(value)
        
        unit = unit.lower()
        if unit in ["billion", "b"]:
            value *= 1000000000
        elif unit in ["million", "m"]:
            value *= 1000000
        
        return value
    
    def calculate_ratios(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculates common financial ratios from extracted metrics.
        
        Args:
            metrics: Dictionary of financial metrics
            
        Returns:
            Dictionary of calculated financial ratios
        """
        ratios = {}
        
        try:
            # Calculate gross margin
            if "revenue" in metrics and "cost_of_goods" in metrics:
                ratios["gross_margin"] = (metrics["revenue"] - metrics["cost_of_goods"]) / metrics["revenue"]
            
            # Calculate net profit margin
            if "revenue" in metrics and "net_income" in metrics:
                ratios["net_profit_margin"] = metrics["net_income"] / metrics["revenue"]
            
            # Calculate P/E ratio
            if "market_cap" in metrics and "net_income" in metrics and metrics["net_income"] > 0:
                ratios["pe_ratio"] = metrics["market_cap"] / metrics["net_income"]
            
            # Calculate ROA (Return on Assets)
            if "net_income" in metrics and "total_assets" in metrics and metrics["total_assets"] > 0:
                ratios["roa"] = metrics["net_income"] / metrics["total_assets"]
            
            # Calculate debt-to-equity ratio
            if "total_liabilities" in metrics and "total_assets" in metrics and "total_liabilities" in metrics:
                equity = metrics["total_assets"] - metrics["total_liabilities"]
                if equity > 0:
                    ratios["debt_to_equity"] = metrics["total_liabilities"] / equity
            
            # Format ratios as percentages where appropriate
            for ratio in ["gross_margin", "net_profit_margin", "roa"]:
                if ratio in ratios:
                    ratios[f"{ratio}_pct"] = f"{ratios[ratio] * 100:.2f}%"
                    
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating ratios: {str(e)}")
            return ratios
    
    def extract_financial_trends(self, text: str) -> Dict[str, Any]:
        """
        Extracts trend information from financial text.
        Looks for growth/decline language and percentage changes.
        
        Args:
            text: Text containing financial trend information
            
        Returns:
            Dictionary of extracted trend data
        """
        trends = {}
        
        try:
            # Look for growth/increase patterns
            growth_pattern = r"(?:increased|grew|rose|up|higher) by (?:approximately |about |roughly )?(\d+(?:\.\d+)?)(?:\s*%)?"
            growth_match = re.search(growth_pattern, text, re.IGNORECASE)
            if growth_match:
                trends["growth"] = float(growth_match.group(1))
                trends["direction"] = "increase"
            
            # Look for decline/decrease patterns
            decline_pattern = r"(?:decreased|declined|fell|down|lower) by (?:approximately |about |roughly )?(\d+(?:\.\d+)?)(?:\s*%)?"
            decline_match = re.search(decline_pattern, text, re.IGNORECASE)
            if decline_match:
                trends["growth"] = -float(decline_match.group(1))
                trends["direction"] = "decrease"
            
            # Look for year-over-year or quarter-over-quarter indicators
            if "YOY" in text or "year-over-year" in text.lower() or "year over year" in text.lower():
                trends["period"] = "YOY"
            elif "QOQ" in text or "quarter-over-quarter" in text.lower() or "quarter over quarter" in text.lower():
                trends["period"] = "QOQ"
                
            return trends
            
        except Exception as e:
            self.logger.error(f"Error extracting trends: {str(e)}")
            return trends 