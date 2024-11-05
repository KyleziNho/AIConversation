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
        # Explicitly check for API keys
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        perplexity_key = os.environ.get("PERPLEXITY_API_KEY")
        
        if not all([anthropic_key, openai_key, perplexity_key]):
            missing_keys = []
            if not anthropic_key: missing_keys.append("ANTHROPIC_API_KEY")
            if not openai_key: missing_keys.append("OPENAI_API_KEY")
            if not perplexity_key: missing_keys.append("PERPLEXITY_API_KEY")
            raise Exception(f"Missing API keys: {', '.join(missing_keys)}")

        return {
            'anthropic': Anthropic(api_key=anthropic_key),
            'openai': OpenAI(api_key=openai_key),
            'perplexity_key': perplexity_key
        }
    except Exception as e:
        logger.error(f"Error initializing API clients: {str(e)}")
        raise

def get_ai_response(clients, prompt, model_type, context=""):
    try:
        logger.info(f"Getting AI response using {model_type}")
        
        if model_type == "anthropic":
            # Fixed prompt formatting for Anthropic
            messages = [
                {
                    "role": "user",
                    "content": f"{prompt}\n\nContext: {context}"
                }
            ]
            
            response = clients['anthropic'].messages.create(
                model="claude-3-sonnet-20240229",
                messages=messages,
                max_tokens=150
            )
            return response.content[0].text
            
        elif model_type == "llama":
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
                    ],
                    "max_tokens_to_sample": 150
                },
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
                
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
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error in get_ai_response: {str(e)}")
        raise Exception(f"API request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error in get_ai_response: {str(e)}")
        raise

def get_perplexity_insights(perplexity_key, topic):
    try:
        logger.info(f"Getting Perplexity insights for topic: {topic}")
        
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
        
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Perplexity API request error: {str(e)}")
        raise Exception(f"Perplexity API request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error in get_perplexity_insights: {str(e)}")
        raise

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/debate', methods=['POST'])
def simulate_debate():
    try:
        # Get topic from JSON body
        data = request.get_json()
        if not data or 'topic' not in data:
            return jsonify({"error": "Missing topic in request"}), 400
            
        topic = data['topic']
        logger.info(f"Starting debate simulation for topic: {topic}")
        
        # Get API clients
        clients = get_api_clients()

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
        return jsonify({
            "error": "Simulation Error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Not found", "path": request.path}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": str(error),
        "timestamp": datetime.utcnow().isoformat()
    }), 500

# Configure for production
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
