from flask import Flask, jsonify, render_template, request, send_from_directory
from datetime import datetime, UTC
import os
import logging
from typing import Dict, List
from dataclasses import dataclass
import requests
import json
from anthropic import Anthropic
from openai import OpenAI

# Initialize Flask app
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExpertAnalysis:
    role: str
    specialty: str
    base_prompt: str
    followup: str
    model: str

class OTAIndustryAnalyst:
    def __init__(self, perplexity_key: str, anthropic_key: str, openai_key: str):
        self.perplexity_key = perplexity_key
        self.anthropic = Anthropic(api_key=anthropic_key)
        self.openai = OpenAI(api_key=openai_key)
        self.headers = {
            "Authorization": f"Bearer {perplexity_key}",
            "Content-Type": "application/json"
        }
        self.compiled_responses = ""
        self.market_scan = ""
        
        self.base_ota_prompt = """As a Market Analyst specializing in Online Travel Agency trends and strategies, analyse latest developments since October 2023.

Task Details:
1. Focus Scope: Only information after October 2023
2. Market Aspects:
   - Political landscape (changing political stability of China and US)
   - Economic Aspects (Global GDP growth and its affects on tourism, global middle class growth)
   - Consumer Behaviour (increase in mobile usage, increase in bleisure travel, extended hotel stay growth, 60+ year old population growth))
   - Technological advancements (Talk about how AI is used in OTAs, personalization)
   - Partnerships, mergers, acquisitions (Focus on recent acquisitions and talk about how Booking.com and Expedia group own majority of OTAs)
   - Regulatory/policy changes (Key legal compliance law changes and potential fines)
3. Information Verification: Corroborate with multiple sources
4. Output: Brief overview + specific examples

Search recent articles and reports, filter for major developments, validate findings, summarize key trends."""

        self.experts = [
            ExpertAnalysis(
                "Consumer Behaviour Expert (GPT-4 + Claude-3)",
                "consumer_trends",
                "Based on the market scan, list 5 key consumer behavior uncertainties in OTAs:",
                "How do economic conditions affect these uncertainties?",
                "multi"
            ),
            ExpertAnalysis(
                "Technology Innovation Specialist (GPT-4 + Claude-3)",
                "tech_innovation",
                "Based on the market scan, list 5 critical technological uncertainties facing OTAs:",
                "Which emerging technologies could disrupt these patterns?",
                "multi"
            ),
            ExpertAnalysis(
                "Strategic Partnership Analyst (GPT-4 + Claude-3)",
                "partnerships",
                "Based on the market scan, list 5 key partnership/consolidation uncertainties:",
                "How might market dynamics affect these trends?",
                "multi"
            ),
            ExpertAnalysis(
                "Regulatory Compliance Specialist (GPT-4 + Claude-3)",
                "regulation",
                "Based on the market scan, list 5 major regulatory uncertainties:",
                "How might policy changes affect these challenges?",
                "multi"
            ),
            ExpertAnalysis(
                "Economic Specialist (GPT-4 + Claude-3)",
                "economics",
                "Based on the market scan, list 5 major economic uncertainties:",
                "How might economic changes affect these challenges?",
                "multi"
            ),
            ExpertAnalysis(
                "Strategic Moderator (GPT-4 + Claude-3)",
                "synthesis",
                "Based on all expert analyses, create strategic plan for Trip.com",
                "Key success factors and roadblocks?",
                "claude"
            )
        ]

    def get_expert_analysis(self, expert: ExpertAnalysis) -> Dict:
        try:
            if not self.market_scan:
                self.market_scan = self._call_perplexity(self.base_ota_prompt)
            
            if expert.specialty != "synthesis":
                base_prompt = f"Market Context: {self.market_scan}\n\n{expert.base_prompt}"
                
                gpt_initial = self._call_gpt(base_prompt)
                claude_initial = self._call_claude(base_prompt)
                
                summary_prompt = f"""Analyze and synthesize these responses:
GPT-4: {gpt_initial}
Claude-3: {claude_initial}

Provide concise summary of key points:"""
                
                initial_response = self._call_claude(summary_prompt)
                
                gpt_followup = self._call_gpt(f"Re: {initial_response}\n{expert.followup}")
                claude_followup = self._call_claude(f"Re: {initial_response}\n{expert.followup}")
                
                followup_prompt = f"""Synthesize follow-up insights:
GPT-4: {gpt_followup}
Claude-3: {claude_followup}

Key points summary:"""
                
                followup_response = self._call_claude(followup_prompt)
                
            else:
                synthesis_prompt = f"""Expert Analyses Summary:
{self.compiled_responses}

Create Trip.com strategic plan:
1. Key insights (bullet points)
2. Strategic priorities (numbered)
3. Implementation timeline"""

                initial_response = self._call_claude(synthesis_prompt)
                followup_response = self._call_claude(f"Based on this plan:\n{initial_response}\n\nList critical success factors and potential roadblocks:")
            
            return {
                "role": expert.role,
                "initial_analysis": initial_response,
                "followup_analysis": followup_response,
                "model_used": "Multi-AI Analysis" if expert.specialty != "synthesis" else "Claude-3 Synthesis"
            }

        except Exception as e:
            logger.error(f"Expert analysis error - {expert.role}: {str(e)}")
            raise

    def _call_claude(self, prompt: str) -> str:
        response = self.anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{
                "role": "user",
                "content": f"{prompt}\n\nProvide clear, concise response."
            }],
            max_tokens=1000
        )
        return response.content[0].text

    def _call_gpt(self, prompt: str) -> str:
        response = self.openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an OTA expert. Provide concise, focused responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content

    def _call_perplexity(self, prompt: str) -> str:
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": "You are analysing OTA industry trends. Provide factual, evidence-based analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def run_analysis(self) -> Dict:
        results = {}
        self.compiled_responses = ""
        
        try:
            # Run expert analyses
            for expert in self.experts[:-1]:
                result = self.get_expert_analysis(expert)
                results[expert.specialty] = result
                self.compiled_responses += f"\n\n{expert.role}:\n{result['initial_analysis']}\nFollow-up: {result['followup_analysis']}"
            
            # Run moderator analysis
            moderator = self.experts[-1]
            results[moderator.specialty] = self.get_expert_analysis(moderator)
            
            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "analyses": results,
                "metadata": {
                    "analysis_period": "Since October 2023",
                    "industry_focus": "OTA",
                    "analysis_type": "Strategic Assessment",
                    "market_scan": "Perplexity AI",
                    "expert_analysis": "Multi-AI (GPT-4 + Claude-3)"
                }
            }
        except Exception as e:
            logger.error(f"Analysis run failed: {str(e)}")
            raise

app = Flask(__name__)

# Catch-all route for serving static files and index
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
   if path == "":
       return render_template('index.html')
   return app.send_static_file(path)

@app.route('/api/analyze', methods=['POST'])
def analyze():
   try:
       data = request.get_json()
       if not data or 'topic' not in data:
           return jsonify({"error": "Missing topic"}), 400

       keys = {
           "perplexity": os.getenv("PERPLEXITY_API_KEY"),
           "anthropic": os.getenv("ANTHROPIC_API_KEY"), 
           "openai": os.getenv("OPENAI_API_KEY")
       }

       if not all(keys.values()):
           missing = [k for k, v in keys.items() if not v]
           return jsonify({"error": f"Missing API keys: {', '.join(missing)}"}), 500

       analyst = OTAIndustryAnalyst(keys["perplexity"], keys["anthropic"], keys["openai"])
       results = analyst.run_analysis()
       return jsonify(results)

   except Exception as e:
       logger.error(f"Analysis failed: {str(e)}")
       return jsonify({
           "error": "Analysis failed",
           "message": str(e),
           "timestamp": datetime.now(UTC).isoformat()
       }), 500

if __name__ == '__main__':
   app.run(port=5010)