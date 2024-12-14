# TrivialAI
_(A set of `requests`-based, trivial bindings for AI models)_

## Basics

```
$ pip install pytrivialai
$ python
>>> from trivialai import claude, gcp, ollama, chatgpt
>>>
```

## Basic model usage

### Ollama

```
>>> client = ollama.Ollama("gemma2:2b", "http://localhost:11434/")
>>> client.generate("This is a test message. Use the word 'platypus' in your response.", "Hello there! :D").content
'Hey!  Did you know platypuses lay eggs and have webbed feet? Pretty cool, huh? 😁'
>>> client.generate_json("This is a test message. Use the word 'platypus' in your response.", "Generate a list of animal names in JSON format. Return [Name] and no other commentary").content
[{'name': 'Platypus'}, {'name': 'Eagle'}, {'name': 'Elephant'}, {'name': 'Giraffe'}, {'name': 'Lion'}, {'name': 'Tiger'}, {'name': 'Zebra'}, {'name': 'Dolphin'}, {'name': 'Whale'}, {'name': 'Bear'}, {'name': 'Wolf'}, {'name': 'Dog'}, {'name': 'Cat'}, {'name': 'Monkey'}]
>>>
```

### Claude

```
>>> client = claude.Claude("claude-3-5-sonnet-20240620", os.environ["ANTHROPIC_API_KEY"])
>>> client.generate("This is a test message. Use the word 'platypus' in your response.", "Hello there! :D").content
"Hello! It's nice to meet you. I hope you're having a fantastic day. Since you mentioned using a specific word, I'll incorporate it here: Did you know that the platypus is one of the few mammals that can produce venom? It's quite an unusual and fascinating creature!"
>>>
```

### GCP

```
>>> client = gcp.GCP("gemini-1.5-flash-001", "/path/to/your/gcp_creds.json", "us-central1")
>>> client.generate("This is a test message. Use the word 'platypus' in your response.", "Hello there! :D").content
"Hello! :D It's great to hear from you. Did you know platypuses are one of the few mammals that lay eggs? 🥚  They are truly fascinating creatures!  What can I help you with today? 😊"
>>> 
```

### ChatGPT

```
>>> client = chatgpt.ChatGPT("gpt-3.5-turbo", os.environ["OPENAI_API_KEY"])
>>> client.generate("This is a test message. Use the word 'platypus' in your response.", "Hello there! :D").content
'Hello! How are you today? By the way, did you know the platypus is one of the few mammals that lays eggs?'
>>> 
```

## Basic Tool Use

```
>>> from src.trivialai import tools
>>> client = ollama.Ollama("gemma2:2b", "http://localhost:11434/")
>>> tls = tools.Tools()
>>> from typing import Optional, List
>>> def _screenshot(url: str, selectors: Optional[List[str]] = None) -> None:
    "Takes a url and an optional list of selectors. Takes a screenshot"
    print(f"GOT {url}, {selectors}!")
... ... ... 
>>> tls.define(_screenshot)
True
## You could also equivalently use
## >>> @tls.define()
## >>> def _screenshot(url: str, selectors: Optional[List[str]] = None) -> None:
##    "Takes a url and an optional list of selectors. Takes a screenshot"
##    print(f"GOT {url}, {selectors}!")
>>> tls.list()
[{'name': '_screenshot', 'type': {'url': <class 'str'>, 'selectors': typing.Optional[typing.List[str]]}, 'description': 'Takes a url and an optional list of selectors. Takes a screenshot'}]
>>> res = client.generate_tool_call(tls, "This space intentionally left blank.",  "Take a screenshot of the Google website and highlight the search box")
>>> res.content
{'functionName': '_screenshot', 'args': {'url': 'https://www.google.com', 'selectors': ['#search']}}
>>> tls.call(res.content)
GOT https://www.google.com, ['#search']!
>>> 
```
