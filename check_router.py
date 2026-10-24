from router import choose_route


CASES = [
    ("What is a Python list?", "direct"),
    ("What is 12 multiplied by 8?", "direct"),
    ("Compare Python lists and tuples.", "direct"),
    ("Does Nimbus Starter include single sign-on?", "standard"),
    ("How long does Nimbus Pro retain audit logs?", "standard"),
    ("Who is the CEO of Nimbus?", "standard"),
    (
        "How many users do Nimbus Starter and Nimbus Pro support?",
        "deep",
    ),
    ("Compare the Nimbus plans.", "deep"),
]


def main():
    for question, expected_route in CASES:
        decision = choose_route(question)
        actual_route = decision["route"]

        assert actual_route == expected_route, (
            f"{question!r}: expected {expected_route}, got {actual_route}"
        )

        print(f"PASS | {actual_route} | {question}")
        print(f"Reason: {decision['reason']}\n")

    try:
        choose_route("   ")
    except ValueError:
        print("PASS | Empty question rejected")
    else:
        raise AssertionError("An empty question should have been rejected.")

    print("All router checks passed.")


if __name__ == "__main__":
    main()