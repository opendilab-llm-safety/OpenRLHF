import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def request_api_wrapper(url, data, score_key="rewards", try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    
    # Ensure data has the required "queries" field
    if "queries" not in data:
        raise ValueError("Missing required field 'queries' in request data")
            
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            return response.get(score_key)
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def remote_rm_fn(api_url, queries, references=None, score_key="rewards"):
    """remote reward model API
    Args:
        api_url: RM API URL
        queries: list of query strings to score
        references: optional list of reference answers
        score_key: key for the score in the response
    Returns:
        torch.Tensor: reward scores
    """
    data = {"queries": queries}
    if references is not None:
        data["references"] = references
        
    scores = request_api_wrapper(api_url, data, score_key)
    # print(f"Remote reward model info:\n data: {data}, scores: {scores}")
    return torch.tensor(scores)


@ray.remote
def remote_rm_fn_ray(api_url, queries, references=None, score_key="rewards"):
    """Ray remote version of remote_rm_fn"""
    return remote_rm_fn(api_url, queries, references, score_key)


if __name__ == "__main__":
    # test utils
    url = "http:xxx/get_reward"
    score = remote_rm_fn(url, queries=["example query"], references=["example response"])
    print(score)
