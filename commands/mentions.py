"""@ 提及解析工具 - 判断消息链里「真正被点名」的用户是谁。

群聊中 @机器人 是唤醒 AstrBot 的常规方式，因此「@机器人 查询好感度」
这类消息的消息链里必然带着一个指向机器人自己的 At 段（OneBot v11 还会
把 @机器人 之后的空格一并发过来）。若直接取消息链中的第一个 At，机器人
就会被当成被查询/被操作对象，于是出现：

    @机器人 查询好感度  →  查到机器人自己的好感度

这里统一忽略「指向机器人自身」与「@全体成员」的 At 段，用户指令与
管理员指令共用同一套判定，避免各指令各写一份、行为不一致。

判定逻辑与 AstrBot 框架自身的唤醒判定保持一致：
框架在 _WakingCheckStage 中同样是用
`str(message.qq) == str(event.get_self_id())` 来识别「@到了机器人自己」。
"""

import astrbot.api.message_components as Comp

# @全体成员 / @所有人 在各平台适配器中的占位 ID
ALL_MENTION_IDS = {"all", "everyone", "allmember", "全体成员"}


def extract_at_target_id(event) -> str | None:
    """返回消息链中第一个「非机器人自身、非全体成员」的 @ 目标用户 ID。

    没有有效的 @ 目标时返回 None，此时调用方应回退为查询发送者自己，
    或改用纯文本参数（如 `查询好感度 123456`）解析。
    """
    self_id = ""
    get_self_id = getattr(event, "get_self_id", None)
    if callable(get_self_id):
        try:
            self_id = str(get_self_id() or "").strip()
        except Exception:
            self_id = ""

    for comp in _iter_message_components(event):
        # 注意：必须做类型判断，OneBot v11 适配器会给 Reply 组件也带上 qq 字段，
        # 用鸭子类型（hasattr(comp, "qq")）会把引用消息误判成 @ 提及。
        if not isinstance(comp, Comp.At):
            continue

        qq = str(getattr(comp, "qq", "") or "").strip()
        if not qq or qq.lower() in ALL_MENTION_IDS:
            continue
        if self_id and qq == self_id:
            # @机器人 只是唤醒前缀，不是查询/操作目标
            continue
        return qq

    return None


def _iter_message_components(event) -> list:
    """兼容取消息链：优先 get_messages()，回退 message_obj.message。"""
    getter = getattr(event, "get_messages", None)
    if callable(getter):
        try:
            return list(getter() or [])
        except Exception:
            return []
    message_obj = getattr(event, "message_obj", None)
    return list(getattr(message_obj, "message", None) or [])
