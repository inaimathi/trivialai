# TrivialAI
_(A set of `requests`-based, trivial bindings for AI models)_

## Basic models

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


