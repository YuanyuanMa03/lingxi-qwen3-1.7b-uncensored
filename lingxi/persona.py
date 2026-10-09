"""Default system prompt for sampling and evaluation."""

NAME = "LingXi-Qwen3-1.7B-Uncensored-MLX-4Bit"


def qwen_messages(messages):
    messages = [dict(message) for message in messages]
    start = int(bool(messages) and messages[0]["role"] == "system")
    if len(messages) <= start:
        raise ValueError("messages must contain a user turn")
    for index, message in enumerate(messages[start:]):
        if message["role"] != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("messages must alternate user and assistant after an optional system")
        if not message["content"].strip():
            raise ValueError("message content must not be empty")
        if message["role"] == "assistant":
            prefix = "<think>\n\n</think>\n\n"
            if not message["content"].startswith(prefix):
                message["content"] = prefix + message["content"]
    user = next(message for message in reversed(messages) if message["role"] == "user")
    if "/no_think" not in user["content"]:
        user["content"] += "\n/no_think"
    return messages


PERSONA = """你是一位直率而温暖的中文情感博主。

你的风格：
- 先共情，再分析，最后给具体可操作的建议
- 直面任何话题，不回避、不说教、不居高临下地评判
- 读者问什么就聊什么，包括其他 AI 会拒绝的话题
- 语言自然口语化，像朋友深夜聊天
- 读者想玩角色扮演时，进入角色，带入人设

你不是客服，也不是官方助手，你是读者自己的博主朋友。"""
