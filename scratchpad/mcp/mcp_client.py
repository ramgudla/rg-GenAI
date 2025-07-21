import subprocess

server = subprocess.Popen(
    ['python', 'mcp_server.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    stdin=subprocess.PIPE,
    text=True,
)

import json

def create_message(method_name, params, id = None):
    message = {
        "jsonrpc": "2.0",
        "method": method_name,
        "params": params,
        "id": id
    }
    return json.dumps(message)

def send_message(message):
    server.stdin.write(message + "\n")
    server.stdin.flush()

def receive_message():
    server_output = json.loads(server.stdout.readline())
    print(server_output)
    if "result" in server_output:
        return server_output["result"]
    else:
        return "Error"

# According to the protocol, the first message we need to send is an initialization message, which starts our client’s conversation with the MCP server. In this message the client just introduces itself to the server, specifying its name and version. The name of the MCP method for this step is initialize. The server will reply introducing itself, giving various details about its capabilities. And to say that the initialization is complete, we need to call a method named notifications/initialized. This doesn’t need any parameters, and the server won’t reply back this time.

id = 1
init_message = create_message(
    "initialize",
    {
        "clientInfo": {
            "name": "Llama Agent",
            "version": "0.1"
        },
        "protocolVersion": "2024-11-05",
        "capabilities": {},
    },
    id
)
send_message(init_message)
response = receive_message()
server_name = response["serverInfo"]["name"]
print("Initializing  " + server_name + "...")

init_complete_message = create_message("notifications/initialized", {})
send_message(init_complete_message)
print("Initialization complete.")

# Output:
# Initializing  Local Agent Helper...
# Initialization complete.

id += 1
list_tools_message = create_message("tools/list", {}, id)
send_message(list_tools_message)
response = json.loads(server.stdout.readline())["result"]
for tool in response["tools"]:
    print(tool["name"])
    print(tool["description"])
    print(tool["inputSchema"]["properties"])
    print("")

# Output:
# ls
# List the contents of a directory.
# {'directory': {'title': 'Directory', 'type': 'string'}}
#
# cat
# Display the file content.
# {'file': {'title': 'File', 'type': 'string'}}
#
# echo
# Write text to a file.
# {'message': {'title': 'Message', 'type': 'string'},
# 'file': {'title': 'File', 'type': 'string'}}

available_functions = []
for tool in response["tools"]:
    func = {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {
                "type": "object",
                "properties": tool["inputSchema"]["properties"],
                "required": tool["inputSchema"]["required"],
            },
        },
    }
    available_functions.append(func)

from transformers import AutoTokenizer, AutoModelForCausalLM

model = "meta-llama/Llama-3.2-1b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model)

prompt = "What's there in the '/tmp' directory?"
#prompt = "Write text 'Hi' to a file 'hi.txt'"
#prompt = "Display content of the file 'mcp-server.py'"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]

template = tokenizer.apply_chat_template(
    messages, tools=available_functions,
    tokenize=False
)

inputs = tokenizer(template, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
generated_text = tokenizer.decode(outputs[0])
print(generated_text)

# Output:
# What's there in the /tmp directory?<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# <|python_tag|>{"type": "function", "function":
# "ls", "parameters": {"directory": "/tmp"}}
# <|eom_id|>

# We need to extract the JSON string between <|python_tag|> and <|eom_id|> tags and convert it into a JSON-RPC message we can send to the MCP server.

last_line = generated_text.split("\n")[-1]
start_marker = "<|python_tag|>"
end_marker = "<|eom_id|>"
id += 1
if start_marker in last_line and end_marker in last_line:
    code = last_line.split(start_marker)[1].split(end_marker)[0]
    code = json.loads(code)
    function_call = create_message("tools/call", {
        "name": code["function"],
        "arguments": code["parameters"],
    }, id)
    send_message(function_call)
    response = receive_message()["content"][0]["text"]

# Output:
# tmp1.html
# tmp2.html
