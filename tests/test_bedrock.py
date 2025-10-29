import os
import json
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError
from dotenv import load_dotenv

# ============================
# Load environment variables
# ============================
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
MODEL_ID = os.getenv("MODEL_ID")

# ============================
# Test Bedrock Connectivity
# ============================
def test_bedrock_connection():
    try:
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

        # Handle Claude 3 (Messages API)
        if MODEL_ID.startswith("anthropic.claude-3"):
            response = client.converse(
                modelId=MODEL_ID,
                messages=[
                    {"role": "user", "content": [{"text": "Hello Claude 3, can you confirm connectivity?"}]}
                ],
                inferenceConfig={
                    "maxTokens": 200,
                    "temperature": 0.7,
                    "topP": 0.9,
                },
            )
            result = response["output"]["message"]["content"][0]["text"]

        # Handle Claude 2 (Legacy Completion API)
        elif MODEL_ID.startswith("anthropic.claude-v2"):
            body = {
                "prompt": "\n\nHuman: Hello Claude 2, can you confirm connectivity?\n\nAssistant:",
                "max_tokens_to_sample": 200,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9,
                "stop_sequences": ["\n\nHuman:"],
            }
            response = client.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json",
            )
            result = json.loads(response["body"].read().decode("utf-8"))["completion"]

        else:
            raise ValueError(f"❌ Unsupported MODEL_ID schema: {MODEL_ID}")

        print("✅ Connection successful! Model response:")
        print(result)

    except NoCredentialsError:
        print("❌ No credentials found. Please check your .env file.")
    except ClientError as e:
        print(f"❌ AWS ClientError: {e}")
    except BotoCoreError as e:
        print(f"❌ BotoCoreError: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    test_bedrock_connection()
