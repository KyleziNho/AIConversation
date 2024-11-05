from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
import os
import logging
from anthropic import Anthropic
from openai import OpenAI
import requests
import json
from datetime import datetime

# Initialize Flask app with proper static folder configuration
app = Flask(__name__, 
            static_url_path='',
            static_folder='templates',
            template_folder='templates')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_api_clients():
    try:
        anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
        
        return {
            'anthropic': anthropic_client,
            'openai': openai_client,
            'perplexity_key': perplexity_api_key
        }
    except Exception as e:
        logger.error(f"Error initializing API clients: {str(e)}")
        return None

def get_ai_response(clients, prompt, model_type, context=""):
    try:
        logger.info(f"Getting AI response using {model_type}")
        
        if model_type == "anthropic":
            response = clients['anthropic'].messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": f"{prompt}\n\nContext: {context}"
                }]
            )
            return response.content[0].text
            
        elif model_type == "llama":
            if not clients.get('perplexity_key'):
                return "Error: Perplexity API key not found"
                
            headers = {
                "Authorization": f"Bearer {clients['perplexity_key']}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json={
                    "model": "llama-3.1-sonar-small-128k-online",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert analyst in the travel industry."
                        },
                        {
                            "role": "user",
                            "content": f"{prompt}\n\nContext: {context}"
                        }
                    ]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                raise Exception(f"Perplexity API error: {response.status_code} - {response.text}")
                
        else:  # OpenAI
            response = clients['openai'].chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert analyst in the travel industry."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nContext: {context}"
                    }
                ],
                max_tokens=1500
            )
            return response.choices[0].message.content
            
    except Exception as e:
        logger.error(f"Error in get_ai_response: {str(e)}")
        return f"Error getting response: {str(e)}"

def get_perplexity_insights(perplexity_key, topic):
    try:
        logger.info(f"Getting Perplexity insights for topic: {topic}")
        
        if not perplexity_key:
            return "Error: Perplexity API key not found"
            
        headers = {
            "Authorization": f"Bearer {perplexity_key}",
            "Content-Type": "application/json"
        }
        
        base_prompt = """As a Market Analyst specializing in Online Travel Agency (OTA) trends and strategies, analyze the latest developments in the OTA market since October 2023.

Focus Areas:
1. Recent consumer behavior trends
2. Technological advancements in major OTA platforms
3. Strategic partnerships and market movements
4. Regulatory and policy changes"""

        topic_prompts = {
            "market_trends": "Focusing specifically on consumer behavior trends and market movements, ",
            "technology": "Focusing specifically on technological advancements and digital innovations, ",
            "sustainability": "Focusing specifically on sustainability initiatives and environmental policies, "
        }

        topic_prefix = topic_prompts.get(topic, "")
        
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a specialized OTA market analyst focusing on recent developments since October 2023."
                },
                {
                    "role": "user",
                    "content": f"{base_prompt}\n\n{topic_prefix}analyze and summarize the latest verified developments in the OTA industry."
                }
            ]
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            error_message = f"Perplexity API error: {response.status_code} - {response.text}"
            logger.error(error_message)
            return f"Error getting research: {error_message}"
            
    except Exception as e:
        logger.error(f"Error in get_perplexity_insights: {str(e)}")
        return f"Error in research: {str(e)}"

# Health check endpoint for Vercel
@app.route('/api/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({"status": "healthy"}), 200

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return str(e), 500

@app.route('/simulate_debate/<topic>', methods=['POST'])
def simulate_debate(topic):
    logger.info(f"Starting debate simulation for topic: {topic}")
    
    try:
        # Get API clients
        clients = get_api_clients()
        if not clients:
            return jsonify({"error": "Failed to initialize API clients"}), 500

        # Get initial research using Perplexity with Llama model
        research_insights = {
            "market_trends": get_perplexity_insights(clients['perplexity_key'], "market_trends"),
            "technology": get_perplexity_insights(clients['perplexity_key'], "technology"),
            "sustainability": get_perplexity_insights(clients['perplexity_key'], "sustainability")
        }
        
        # Initialize the analysis chain
        analysis_chain = []
        current_context = json.dumps(research_insights, indent=2)
        
        # Analysis steps with different AI models
        analysis_steps = [
            ("TRAVEL INDUSTRY ANALYST", "openai", "Analyze the market dynamics and industry trends."),
            ("TECHNOLOGY INNOVATION SPECIALIST", "llama", "Evaluate technological implications and future innovations."),
            ("CONSUMER BEHAVIOR RESEARCHER", "anthropic", "Analyze consumer preferences and behavioral shifts."),
            ("SUSTAINABILITY EXPERT", "llama", "Assess environmental impact and sustainability opportunities."),
            ("ECONOMIC STRATEGIST", "openai", "Evaluate economic implications and business models."),
            ("STRATEGIC FORESIGHT MODERATOR", "anthropic", "Synthesize all perspectives and provide strategic recommendations.")
        ]
        
        for expert, model, instruction in analysis_steps:
            prompt = f"""As a {expert}, {instruction}
            Topic: {topic}
            Previous Analysis: {current_context}
            Please provide a detailed analysis from your expertise."""
            
            response = get_ai_response(clients, prompt, model, current_context)
            analysis_chain.append({
                "expert": expert,
                "contribution": response,
                "model": model
            })
            current_context = response
        
        result = {
            "research_insights": research_insights,
            "analysis_chain": analysis_chain
        }
        
        logger.info("Debate simulation completed successfully")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in simulate_debate: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Configure for production
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['JSON_SORT_KEYS'] = False

# For local development only
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)