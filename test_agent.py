from agent import get_competitor_insights


def run_tests():
    print("--- TEST 1: Asking about Static/Existing Info ---")
    q1 = "What are Snyk's advantages in Contextual Analysis?"
    res1, _ = get_competitor_insights(q1, "Snyk")
    print(f"\nFinal Answer 1:\n{res1}\n")
    
    # print("="*50)
    
    print("--- TEST 2: Asking about recent 2026 news (Should trigger Web Search) ---")
    q2 = "What are the latest security features or product announcements Snyk made recently?"
    res2, _ = get_competitor_insights(q2, "Snyk")
    print(f"\nFinal Answer 2:\n{res2}\n")

    print("--- TEST 3: Global Market Query (All Competitors) ---")
    q3 = "What are the latest security features or product announcements across competitors?"
    res3, _ = get_competitor_insights(q3, "All Competitors")
    print(f"\nFinal Answer 3:\n{res3}\n")


if __name__ == "__main__":
    run_tests()