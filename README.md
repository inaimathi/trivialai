# TrivialAI
_(A set of `requests`-based, trivial bindings for AI models)_

## Basic models

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

### Ollama

```
>>> client = ollama.Ollama("gemma2:2b", "http://localhost:11434/")
>>> client.generate("This is a test message. Use the word 'platypus' in your response.", "Hello there! :D").content
'Hey!  Did you know platypuses lay eggs and have webbed feet? Pretty cool, huh? 😁'
>>> 
```

