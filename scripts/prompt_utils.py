def build_prompt(userInput: str) -> str:
    return (
        "### Instruction:\n"
        "Extract ALL gate-size connectivity rules from the input. "
        "Output ONLY valid JSON matching this schema: "
        "{\"rules\":[{\"nQubits\":int,\"shape\":string,\"edges\":[[int,int],...]},...]}.\n"
        "Use canonical node ids 0..k-1 for each rule.\n"
        "### Input:\n"
        f"{userInput}\n"
        "### Output:\n"
    )