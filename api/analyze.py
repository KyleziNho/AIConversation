from http.client import HTTPResponse
import json
from datetime import datetime, UTC
from azure.functions import HttpRequest, HttpResponse

async def main(req: HttpRequest) -> HttpResponse:
    try:
        data = req.get_json()
        
        # Initialize analyst and run analysis with increased timeouts
        analyst = OTAIndustryAnalyst(keys["perplexity"], keys["anthropic"], keys["openai"])
        results = await analyst.run_analysis()  # Make sure this is async
        
        return HttpResponse(
            json.dumps(results),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        return HttpResponse(
            json.dumps({
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat()
            }),
            mimetype="application/json",
            status_code=500
        )