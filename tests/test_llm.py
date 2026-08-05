from app.services.llm import get_llm


def main() -> None:
    llm = get_llm()

    response = llm.invoke(
        "Explain RAG in two simple sentences."
    )

    print("Response object:")
    print(response)

    print("\nGenerated text:")
    print(response.content)


if __name__ == "__main__":
    main()