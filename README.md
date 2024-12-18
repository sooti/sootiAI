# SootiAI

SootiAI is a multi-purpose large language model (LLM) agent designed to perform general tasks on both local machines and online environments. It is simple to use, highly flexible, and equipped with a range of tools to handle tasks like research, data analysis, local file operations, and more.

## Features

1. **Researching Topics**
   - Search and scrape multiple sources to gather information.
   - Generate summaries or detailed research papers with structured sections like Abstract and Results.


2. **Data Handling and Visualization**
   - Gather data online and use it to create data sheets or plot charts using Python.
   - Example: "Plot a graph of the weather in NYC, Chicago, and Houston for the next 3 days."
   - 
   ![canada_rocks](https://github.com/user-attachments/assets/956924de-e5f8-408f-920e-15a5b4cb0448)
   ![Screenshot 2024-12-17 at 14 32 00](https://github.com/user-attachments/assets/6b7e68f5-8a71-46b7-b333-37c1a39c4646)

3. **Local Machine Operations**
   - Execute tasks like creating folders, listing directory contents, or downloading files.
   - Example: "Download the top 3 math PDFs to my home directory under 'math' and sort them by date."

4. **Multi-Tasking**
   - Perform multiple tasks in a single command seamlessly.

5. **User-Friendly Interfaces**
   - **CLI**: Ideal for terminal enthusiasts.
   - **WebUI**: Includes a browser-based interface with local conversation context saving (until cleared). Multi-session save/load functionality is on the roadmap.

## Why SootiAI?

Existing agents often come with limitations such as:
- Complex setup processes.
- Lack of essential tools like scraping and searching.
- Dependence on paid APIs for basic functionalities.
- Inability to write and execute code effectively.
- Poor performance with smaller models or overly complex workflows for simple tasks.

SootiAI bridges these gaps by providing a streamlined, efficient, and flexible solution for users.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/sooti/sootiAI.git
   ```

2. Navigate to the project directory:
   ```bash
   cd sootiAI
   ```

3. Set up a virtual environment:
   ```bash
   python3 -m venv .venv
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Configure the environment:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file to customize the following:
     - **OpenAI Endpoint**: Set the endpoint to local, remote, llama.cpp, or another compatible source.
     - **API Key**: Add an API key if required (not needed for local models).
     - **Model Name**: Specify the model name (e.g., required for MLX, not for llama.cpp).

6. Start the application:
   - For WebUI (default port: 8080):
     ```bash
     python main.py
     ```
   - For CLI mode:
     ```bash
     python main_cli.py
     ```

## Examples of Use Cases

1. **Research and Summarization**
   - "Research the history of quantum computing and summarize it in a research paper format."

2. **Data Visualization**
   - "Plot a line graph showing the temperature trends in San Francisco over the past week."

3. **Local File Operations**
   - "Create a folder named 'Projects' and move all files with '.py' extension into it."

4. **Automated Data Collection**
   - "Scrape the latest stock prices for Apple, Google, and Tesla and save them in a CSV file."

## Recommended Local Models
1. Qwen 14B - I've found this one to be the best balance of speed, script writing and instruction following.
2. Qwen 7B - Good for some basic research cases and basic tasks, but not complex programming requests.
3. EXAONE 7.8B - Good for research, OK for programming tasks.

## Bad models in my tests
1. Llama-3.1
2. Heremes 2 and Hermes 3
3. Gemma 9b - mixed results, sometimes ok but other times fails to follow instructions.

## Roadmap

- Add support for multi-session save/load in the WebUI.
- Enhance CLI commands with more intuitive shortcuts.
- Expand compatibility with additional LLM backends and endpoints.
- Improve documentation and add community-contributed examples.

## Contributing

We welcome contributions! Feel free to open issues or submit pull requests to help improve SootiAI. Make sure to follow the [contributing guidelines](CONTRIBUTING.md) (to be added soon).

## License

SootiAI is licensed under the [MIT License](LICENSE).

---

Feel free to explore and enjoy the capabilities of SootiAI!
