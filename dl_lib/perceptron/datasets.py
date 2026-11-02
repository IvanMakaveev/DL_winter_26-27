def create_dataset(n: int) -> list[tuple[float, float]]:
    return [(float(x), float(2 * x)) for x in range(n)]
