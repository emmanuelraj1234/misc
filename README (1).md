```bash
services:
  service-name:
    image: vllm-openai:v0.11.0
    runtime: nvidia
    ports:
    - "8010:8000"
    volumes:
    - /data/vllm/models/:/models/
    - /data/vllm/tool_chat_template_gemma3_pythonic.jinja:/templates/tool_chat_template_gemma3_pythonic.jinja
    environment: 
    - HF_HUB_OFFLINE=1
    - VLLM_LOGGING_LEVEL=DEBUG
    ipc: host
    deploy:
      ressources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    command: --model /models/google/gemma-3-27b-it --served-model-name google/gemma-3-27b-it --enable-auto-tool-choice --tool-call-parser pythonic --limit-mm-per-prompt '{"image":2}' --gpu-memory-utilization 0.95 --chat-template /templates/tool_chat_template_gemma3_pythonic.jinja
```

```bash
docker cp $(docker create --name scratch docker_user/model:tag none):/data/. ./ && docker rm scratch
```

# template
```
{#- Begin-of-sequence token to start the model prompt -#}
{{ bos_token }}
{#- Extracts the system message. Gemma does not support system messages so it will be prepended to first user message. -#}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{#- Set tools to none if not defined for this ChatCompletion request (helps avoid errors later) -#}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}
{#- Validate alternating user/assistant messages (excluding 'tool' messages and ones with tool_calls) -#}
{%- for message in loop_messages | rejectattr("role", "equalto", "tool") | selectattr("tool_calls", "undefined") -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
{%- endfor -%}

{#- Main loop over all messages in the conversation history -#}
{%- for message in loop_messages -%}
    {#- Normalize roles for model prompt formatting -#}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- elif (message['role'] == 'tool') -%}
        {%- set role = "user" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {#- Mark the start of a message block with the appropriate role -#}
    {{ '<start_of_turn>' + role + '\n' -}}

    {#- Insert system message content (if present) at the beginning of the first message. -#}
    {%- if loop.first -%}
        {{ first_user_prefix }}
        {#- Append system message with tool information if using tools in message request. -#}
        {%- if tools is not none -%}
            {{- "Tools (functions) are available. If you decide to invoke one or more of the tools, you must respond with a python list of the function calls.\n" -}}
            {{- "Example Format: [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)] \n" -}}
            {{- "Do not use variables. DO NOT USE MARKDOWN SYNTAX. You SHOULD NOT include any other text in the response if you call a function. If none of the functions can be used, point it out. If you lack the parameters required by the function, also point it out.\n" -}}
            {{- "Here is a list of functions in JSON format that you can invoke.\n" -}}
            {{- tools | tojson(indent=4) -}}
            {{- "\n\n" -}}
        {%- endif -%}
    {%- endif -%}

    {#- Format model tool calls (turns where model indicates they want to call a tool) -#}
    {%- if 'tool_calls' in message -%}
        {#- Opening bracket for tool call list. -#}
        {{- '[' -}}
        {#- For each tool call -#}
        {%- for tool_call in message.tool_calls -%}
            {#- Get tool call function. -#}
            {%- if tool_call.function is defined -%}
                {%- set tool_call = tool_call.function -%}
            {%- endif -%}
            {#- Function name & opening parenthesis. -#}
            {{- tool_call.name + '(' -}}

            {#-- Handle arguments as list (positional) or dict (named) --#}
            {#-- Named arguments (dict) --#}
            {%- if tool_call.arguments is iterable and tool_call.arguments is mapping -%}
                {%- set first = true -%}
                {%- for key, val in tool_call.arguments.items() -%}
                    {%- if not first %}, {% endif -%}
                    {{ key }}={{ val | tojson }}
                    {%- set first = false -%}
                {%- endfor -%}
            {#-- Positional arguments (list) --#}
            {%- elif tool_call.arguments is iterable -%}
                {{- tool_call.arguments | map('tojson') | join(', ') -}}
            {#-- Fallback: single positional value --#}
            {%- else -%}
                {{- tool_call.arguments | tojson -}}
            {#-- Closing parenthesis. --#}
            {%- endif -%}
                {{- ')' -}}
            {#-- If more than one tool call, place comma and move to formatting next tool call --#}
            {%- if not loop.last -%}, {% endif -%}
        {%- endfor -%}
        {#- Closing bracket for tool call list. -#}
        {{- ']' -}}
    {%- endif -%}
    
    {#- Tool response start tag (for messages from a tool) -#}
    {%- if (message['role'] == 'tool') -%}
        {{ '<tool_response>\n' -}}
    {%- endif -%}

    {#- Render the message content: handle plain string or multimodal content like image/text -#}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}

    {#- Tool response end tag -#}
    {%- if (message['role'] == 'tool') -%}
        {{ '</tool_response>' -}}
    {%- endif -%}

    {#- Mark end of a single turn -#}
    {{ '<end_of_turn>\n' }}
{%- endfor -%}

{#- If generation is to be triggered, add model prompt prefix -#}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\n'}}
{%- endif -%}
```

Template source: https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_gemma3_pythonic.jinja 

# Embedding

```bash
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOEN=<token>" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model "intfloat/multilingual-e5-large-instruct" --task "embed"
```

Call: [https://docs.vllm.ai/en/v0.7.3/getting_started/examples/openai_embedding_client.html](https://docs.vllm.ai/en/v0.7.0/getting_started/examples/openai_embedding_client.html) 
```python
# SPDX-License-Identifier: Apache-2.0

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

responses = client.embeddings.create(
    input=[
        "Hello my name is",
        "The best thing about vLLM is that it supports many different models"
    ],
    model=model,
)

for data in responses.data:
    print(data.embedding)  # list of float of len 4096
```

