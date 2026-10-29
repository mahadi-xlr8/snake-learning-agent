from agent.replay_memory import ReplayMemory


def test_push():
    memory = ReplayMemory(capacity=5)

    memory.push("s1", 0, 10, "s2", False)

    assert len(memory) == 1


def test_capacity():
    memory = ReplayMemory(capacity=2)

    for i in range(3):
        memory.push(i, 0, 0, i + 1, False)

    assert len(memory) == 2
    assert memory.memory[0][0] == 1


def test_sample():
    memory = ReplayMemory()

    for i in range(10):
        memory.push(i, 0, 0, i + 1, False)

    batch = memory.sample(4)

    assert len(batch) == 4