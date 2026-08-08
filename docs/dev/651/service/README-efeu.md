# A local coding assistant on this laptop

You have a coding agent (`omp`) wired to a language model that runs entirely on
this machine's own GPU. Nothing you type goes to the internet.

## Read this first: the agent does not work yet

**Status: not usable.** Everything is installed and wired correctly, but a real
coding request still fails on this laptop's GPU. Short questions to the model
work; the agent does not. Details below.

`omp` is installed and correctly pointed at the local model, but a real coding
request currently **crashes the GPU** and does not return an answer. This is a
firmware fault in this laptop's graphics chip, not a setup mistake, and it is
not something you can configure around.

The short version: this GPU is fine at *generating* text and fails at
*reading* a lot of text at once. A coding agent sends thousands of tokens of
context with every request, which is exactly the operation that fails. You
will see the request hang and then fail, and `dmesg` will show
`GPU reset ... device wedged`. The machine recovers on its own and the service
restarts itself, so nothing is damaged -- but you do not get an answer.

Short questions asked directly of the model DO work. If you want to try it:

```
curl -s localhost:31651/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-35b-a3b",
       "messages":[{"role":"user","content":"What is 6 times 7?"}],
       "chat_template_kwargs":{"enable_thinking":false},
       "max_tokens":64}'
```

The rest of this file describes how the setup works, so it is ready the moment
the underlying fault is fixed.

## Using it

```
omp --model local/qwen36-35b-a3b
```

Run it from inside the project you want to work on. Add `omp` to your PATH
first if the shell cannot find it -- `~/.local/bin` is already appended in
`~/.profile`, so a fresh login shell has it.

### Keep the tool surface small

This model has an 8192-token context, and omp's full built-in tool set does
not fit in it: with all 32 tools loaded the system prompt alone measures
**17029 tokens**, and the server rejects the request before it starts

    400 The input (17029 tokens) is longer than the model's context length (8192 tokens)

That is a property of the machine, not a misconfiguration -- the context is
bounded by RAM the weights are already using. Ask for the tools you need:

```
omp --model local/qwen36-35b-a3b --no-lsp --no-skills --no-extensions \
    --tools=read,write
```

Add tools back one at a time if you need them. If a request fails with the
error above, the answer is always fewer tools or a shorter conversation --
`omp` starts a fresh session with `--no-session` if a long history is what
pushed it over.

## The first question takes about two and a half minutes

This is expected, and it is not a hang.

The model is 21.6 GB and, while loaded, it occupies roughly 22.7 GB of this
laptop's 29.5 GB of memory -- effectively the whole machine. Keeping it loaded
all the time would leave you a laptop you could not use for anything else. So
it is kept unloaded, and loaded on demand:

- You ask a question. The service starts loading the model and **holds your
  request** until it can answer. That first answer takes ~2.5 minutes.
- Every following question answers immediately.
- After **60 seconds** with no activity, the model is unloaded and your memory
  is given back. The next question pays the loading time again.

If you are about to do a stretch of work, the practical trick is simply to ask
something trivial first ("hi") and let it load while you get set up.

### Changing the idle window

If 60 seconds is too eager, raise it:

```
sudo systemctl edit htsglang-ondemand
```

and add:

```
[Service]
Environment=HTSGLANG_IDLE_PARK_SECONDS=600
```

then `sudo systemctl restart htsglang-ondemand`. Ten minutes keeps the model
resident through short pauses in a working session, at the cost of holding your
RAM for that long after you stop.

## Checking on it

```
curl -s localhost:31651/ondemand/status
```

Reports whether the model is `parked`, `loading`, or `up`, how long it has been
idle, and how long the last load took. Asking this does **not** wake the model.

```
systemctl status htsglang-ondemand
journalctl -u htsglang-ondemand -f
```

## If something looks wrong

The service restarts itself on failure. If answers stop coming, the useful
first check is whether the model is loading rather than broken -- the status
endpoint above distinguishes those. The GPU on this machine has a known
firmware fault under heavy prefill (documented as the "MES wedge"); the service
runs with the mitigations for it applied on every single load, but if the
screen or GPU misbehaves after a long request, `dmesg | grep -i "GPU reset"`
will say so plainly.


## Which model file is served, and why not a smaller one

The service runs `Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf` (22.7 GB). That file was
specially rebuilt for this machine to remove tensor types its GPU driver
mishandles, and it is the only checkpoint here that produces correct output.

There is a much smaller file on the machine (`Q2_K_XL`, 11.7 GB) which would
leave far more memory free. **It must not be used.** It contains tensor types
whose arithmetic on this GPU produces non-finite garbage, and it crashes on
startup. Smaller is the right instinct for this laptop, but it needs a
correctly rebuilt file, not this one.

To change the checkpoint, edit `MODEL=` in
`/etc/systemd/system/htsglang-ondemand.service.d/20-model.conf` and
`sudo systemctl restart htsglang-ondemand`.
