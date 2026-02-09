from groq import Groq
from typing import Any, Callable
import threading
import time
import uuid
import json
from RAG_1.RAG.helpers import extract_first_json

# Ai_Assistant w
class Ai_Assistant:
    def __init__(self, max_messages: int = 16):
        self.max_messages = max_messages
        # each item: {"role":"user"/"assistant"/"tool"/"system"/"tool_request"/"assistant_raw", "text": "...", ...}
        self.history: list[dict] = []
        self.client = Groq()

        # tools registry: name -> callable(query: str) -> result
        self.tools: dict[str, Callable[[str], Any]] = {}
        # per-tool timeout (seconds) for the threaded wrapper
        self.tool_timeout: float = 8.0

        self.system_prompt = (
            """ You are a DREAM INTERPRETATION assistant.  
Your goal is to interpret the user's dreams with clarity, symbolic insight, and psychological depth — without fiction, hallucination, or mystical claims.

You have access to two tools:

1. search_internet
- Use this only when the dream requires information about symbols, mythology, cultural meanings, or current psychological theory that you are not certain about.

2. search_doc
- Use this only to search the vector database of the user's dream notes.

When a tool is needed, you must return ONLY a JSON object in the following format:

{
"tool": "<tool_name>",
"query": "<your query>"
}

No text should appear before or after this JSON object.

When no tool call is required, interpret the dream in normal text.  
Provide symbolic meaning, emotional analysis, and patterns across dreams if relevant. Keep iterpretation cleear of any JSON

If you need to add commentary that is not part of the JSON tool call, keep it outside the JSON object.

At last, when you have gathered enough information from tools, provide a final interpretation in normal text.

ONLY RETURN THE JSON WHEN CALLING A TOOL.
NO JSON SHOULD BE RETURNED IN THE FINAL INTERPRETATION. 
"""
        )

    # tool registration
    def register_tool(self, name: str, fn: Callable[[str], Any]) -> None:
        if name in self.tools:
            raise ValueError(f"tool '{name}' already registered")
        self.tools[name] = fn

    # safe tool runner (threaded timeout) 
    async def _run_tool(self, tool_name: str, query: str) -> dict:
        """
        Run a registered tool with a timeout. Returns {"ok":bool, "output":..., "error":...}
        """
        print(f"_run_tool: Invoking tool '{tool_name}' with query: {query}")
        if tool_name not in self.tools:
            return {"ok": False, "output": None, "error": f"unknown tool '{tool_name}'"}

        result = {"ok": False, "output": None, "error": None}

        def target():
            try:
                out = self.tools[tool_name](query)
                result["output"] = out
                result["ok"] = True
            except Exception as e:
                result["error"] = str(e)
                result["ok"] = False

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        # wait for completion or timeout
        thread.join(self.tool_timeout)
        #if still alive after timeout, report timeout
        if thread.is_alive():
            result["ok"] = False
            result["error"] = f"tool '{tool_name}' timed out after {self.tool_timeout}s"
        return result

    # parse tool instruction 
    def _parse_tool_instruction(self, llm_text: str) -> dict | None:
        """
        Use extract_first_json to find a JSON object inside llm_text. If it contains
        'tool' and 'query' and the tool is registered, return the dict; else None.
        """
        instr = extract_first_json(llm_text)
        if not instr or not isinstance(instr, dict):
            return None
        if "tool" in instr and "query" in instr and instr["tool"] in self.tools:
            return {"tool": instr["tool"], "query": instr["query"]}
        return None

    # history management 
    def add_user(self, text: str):
        self.history.append({"role": "user", "text": text})
        self._trim()
        print(f"add_user: \nItem added: {self.history[-1]}\nHistory length:", len(self.history))

    def add_assistant(self, text: str):
        self.history.append({"role": "assistant", "text": text})
        self._trim()

    def add_tool(self, text: str):
        #we should keep tool entries separately, [KISS]
        self.history.append({"role": "tool", "text": text})
        self._trim()

    def _trim(self, merge_threshold: int = 3):
        """
        Enforce history window and perform adaptive merging of an existing summary with
        some of the last-6 messages when appropriate.
        """

        # only trim if we have exceeded allowed messages
        if len(self.history) <= self.max_messages - 1:
            print(f"_trim: No trimming needed.\nHistory length:", len(self.history))
            return

        # newest window
        window = self.history[-self.max_messages:]
        if len(window) <= 6:
            self.history = window
            return

        compressible = window[:-6]
        last6 = window[-6:]

        # if compressible is not a summary item → make a new summary
        if not (len(compressible) == 1 and compressible[0].get("type") == "summary"):
            try:
                summary_text = self._summarise_messages(compressible)
            except Exception:
                self.history = window
                return

            prev_version = 0
            if len(self.history) >= 7 and self.history[-7].get("type") == "summary":
                prev_version = self.history[-7].get("meta", {}).get("version", 0)

            summary_item = {
                "role": "system",
                "type": "summary",
                "text": summary_text,
                "meta": {
                    "compressed_count": len(compressible),
                    "timestamp": time.time(),
                    "version": prev_version + 1,
                },
            }

            self.history = [summary_item] + last6
            return

        # === compressible IS already a summary ===
        existing_summary = compressible[0]

        if merge_threshold < 0:
            merge_threshold = 0
        if merge_threshold > 6:
            merge_threshold = 6

        mergeable_count = max(0, len(last6) - merge_threshold)
        if mergeable_count <= 0:
            self.history = [existing_summary] + last6
            return

        to_merge = []
        to_merge.append({
            "role": existing_summary.get("role", "system"),
            "text": existing_summary.get("text", "")
        })
        to_merge.extend(last6[:mergeable_count])

        try:
            new_summary_text = self._summarise_messages(to_merge)
        except Exception:
            self.history = window
            return

        prev_version = existing_summary.get("meta", {}).get("version", 0)
        new_version = prev_version + 1

        new_summary_item = {
            "role": "system",
            "type": "summary",
            "text": new_summary_text,
            "meta": {
                "compressed_count": existing_summary.get("meta", {}).get("compressed_count", 0)
                                    + mergeable_count,
                "timestamp": time.time(),
                "version": new_version,
                "merged_last_count": mergeable_count,
            },
        }

        remaining_tail = last6[mergeable_count:]
        self.history = [new_summary_item] + remaining_tail

    def get_existing_summary_text(self) -> str:
        """
        Returns the summary text if the summary item is in the expected position
        (7th from the end) AND is marked with type='summary'.
        Otherwise returns an empty string.
        """
        if len(self.history) >= 7:
            item = self.history[-7]
            if item.get("role") == "system" and item.get("type") == "summary":
                return item.get("text", "")
        return ""

    def as_last_6_context(self, max_chars: int = 800) -> str:
        """
        Return the last up to 6 messages as a single string, bounded by max_chars.
        Keeps the newest messages — if the total text would exceed max_chars,
        it drops older ones first.
        """
        last_msgs = self.history[-6:]  # newest 6 messages
        pieces = []
        total_len = 0

        # iterate newest → oldest (reverse)
        for m in reversed(last_msgs):
            piece = f"[{m['role']}] {m.get('text', '')}"
            piece_len = len(piece)

            if total_len + piece_len > max_chars:
                break

            pieces.append(piece)
            total_len += piece_len

        # reverse back to chronological order
        return "\n".join(reversed(pieces))


    def _summarise_messages(self, messages: list[dict]) -> str:
        if not messages:
            return ""
        context = "\n".join(f"[{m['role']}] {m['text']}" for m in messages)
        system_prompt = (
            "You are a helpful AI assistant. Summarise the following conversation history into concise "
            "bullet points that capture the key information exchanged. Focus on important facts, "
            "decisions, and questions raised. Omit pleasantries and filler content. Format the summary "
            "as a list of bullet points."
        )
        query = f"Conversation History:\n{context}"
        return self.call_LLM(system_prompt, query)

    # call_LLM
    def call_LLM(self, system_prompt: str, query: str) -> str:
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content



    async def ask(self, user_input: str, max_tool_cycles: int = 10) -> str:
        self.add_user(user_input)

        formatted_query = (
            f"{self.system_prompt}\n\n"
            f"Recent Conversation History:\n\n{self.as_last_6_context(max_chars=1000).strip()}"
        )
        if self.get_existing_summary_text().strip():
            formatted_query += f"\n\nOld Summarised History:\n{self.get_existing_summary_text().strip()}"
        formatted_query += f"\nUser Input:\n{user_input}\n"

        llm_reply = ""
        for cycle in range(max_tool_cycles):
            # Step A: if we don't have an LLM answer for the current formatted_query, request it
            if not llm_reply:
                try:
                    llm_reply = self.call_LLM(self.system_prompt, formatted_query)
                except Exception as e:
                    error_text = f"[LLM_ERROR] {str(e)}"
                    self.history.append({"role":"assistant","text":error_text,"meta":{"error":True,"ts":time.time()}})
                    return error_text

                self.history.append({
                    "role": "assistant_raw",
                    "text": llm_reply,
                    "meta": {"cycle_id": str(uuid.uuid4()), "ts": time.time()}
                })

            # Step B: parse LLM output for tool instruction
            instr = self._parse_tool_instruction(llm_reply)
            if instr is None:
                # final answer from LLM
                final_text = llm_reply.strip()
                parsed_json = extract_first_json(final_text)
                if parsed_json:
                    final_text = final_text.replace(json.dumps(parsed_json), "").strip()
                self.history.append({"role": "assistant", "text": final_text})
                return final_text

            # Step C: we have a tool instruction -> run tool synchronously
            tool_name = instr["tool"]
            tool_query = instr["query"]
            self.history.append({"role": "tool_request", "tool": tool_name, "text": tool_query})

            tool_res = await self._run_tool(tool_name, tool_query)  # this must be synchronous or awaited
            print(f"\n\n\n\nask: Tool '{tool_name}' returned: {tool_res}\n\n\n\n")
            if tool_res.get("ok"):
                out = tool_res.get("output")
                out_text = json.dumps(out, ensure_ascii=False) if isinstance(out, (dict, list)) else str(out)
                out_text = out_text.replace("\n", " ")[:3000]
                tool_output_item = {"role": "tool", "tool": tool_name, "text": out_text}
            else:
                tool_output_item = {"role": "tool", "tool": tool_name, "text": ""}

            self.history.append(tool_output_item)

            # Step D: append tool result to formatted query for next LLM call
            formatted_query += f"\n\n[{tool_name}] query: {tool_query}\nresult: {tool_output_item['text']}\n"

            # IMPORTANT: clear llm_reply so the loop will call the LLM with the updated formatted_query
            llm_reply = ""
        # if we exhaust cycles
        return "[ERROR] max_tool_cycles reached"
