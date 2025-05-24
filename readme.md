# cortex 
cortex is an LLM-based robotic task planner capable of processing natural language commands and producing a sequential plan of primitive actions. 

## Setup
Make a workspace folder and clone this repo in the src folder 

```bash 
$ mkdir -p llm_ws/src 
$ cd llm_ws/src 
```

Clone the repo and its submodules 

```bash
$ git clone --recursive https://github.com/samlovelace/cortex.git
``` 

## Building 
```bash 
$ cd llm_ws
$ colcon build  
```

## Configuration 
Download and desired model from HuggingFace or other online source.

Supported model formats 
 - GGUF 

# Config file 
Set the *engine* config to the desired LLM inference engine. Current supported inference engines are 
 - Llama.cpp

Set the *path* config to the folder container the model file 
Set the *model* config to the full model name including file extension

Set the *promptHeader* config to a string of text to be prepended to the natural language command. 
Ideally, this is used for defining the available primitive actions of the robot and doing single-shot or few-shot prompting. 

## Running 
```bash
$ ros2 run cortex cortex
``` 

With the cortex process up and running, send prompts to the LLM via a ros2 topic */cortex/prompt* 

An example prompt sent from the terminal 

```bash 
$ ros2 topic pub /cortex/prompt std_msgs/msg/String "{data: 'clean up this mess'}"
```
