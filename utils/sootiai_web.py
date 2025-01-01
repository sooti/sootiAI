import datetime
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, Set, Generator
from urllib.parse import urlparse
import pkg_resources

import colorama
import stealth_requests as requests
import urllib3
import yt_dlp
from bs4 import BeautifulSoup
from flask_socketio import emit
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth
from tqdm import tqdm
from dotenv import load_dotenv
import os


colorama.init(autoreset=True)
urllib3.disable_warnings()
# Load .env file
load_dotenv()

# Get values from .env
base_model = os.getenv("BASE_MODEL")
base_api = os.getenv("BASE_API")
base_url = os.getenv("BASE_URL")
temperature = float(os.getenv("TEMPERATURE", 0.3))
top_p = float(os.getenv("TOP_P", 1))
frequency_penalty = float(os.getenv("FREQUENCY_PENALTY", 0))
presence_penalty = float(os.getenv("PRESENCE_PENALTY", 0))
max_tokens = int(os.getenv("MAX_TOKENS", 2048))
max_context = int(os.getenv("MAX_CONTEXT", 32000))


def get_installed_packages() -> Set[str]:
    return {pkg.key for pkg in pkg_resources.working_set}


def install_missing_packages(required_packages: Set[str]) -> Dict[str, bool]:
    installed_packages = get_installed_packages()
    missing_packages = required_packages - installed_packages
    results = {}

    for package in missing_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            results[package] = True
        except subprocess.CalledProcessError:
            results[package] = False

    return results


def extract_imports(code: str) -> Set[str]:
    import_pattern = re.compile(r'^(?:from\s+(\S+?)(?:.\S+)?\s+import\s+.*$|import\s+(\S+))', re.MULTILINE)
    matches = import_pattern.finditer(code)
    packages = set()

    for match in matches:
        package = match.group(1) or match.group(2)
        base_package = package.split('.')[0]
        if base_package not in sys.stdlib_module_names:
            packages.add(base_package)

    return packages


# Global variables
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}
RATE_LIMIT = 1
TIMEOUT = 10
MAX_RETRIES = 3
last_request_time = {}


def respect_rate_limit(url):
    domain = urlparse(url).netloc
    current_time = time.time()
    if domain in last_request_time:
        time_since_last_request = current_time - last_request_time[domain]
        if time_since_last_request < RATE_LIMIT:
            time.sleep(RATE_LIMIT - time_since_last_request)
    last_request_time[domain] = current_time


def scrape_website(url: str) -> dict[str, dict[str, Any] | str] | str:
    try:
        # Respect rate limits
        respect_rate_limit(url)

        # Fetch and parse response
        response = requests.get(url, verify=False, headers=HEADERS, impersonate='safari')
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unnecessary elements
        for tag in soup(['nav', 'footer', 'aside', 'script', 'style']):
            tag.decompose()

        # Extract and clean text
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces/newlines into one
        text = re.sub(r'[^\w\s.,?!]', '', text)  # Remove special characters
        text = " ".join(text.split()[:int(max_context / 8)])  # Truncate to fit context size

        # Extract and clean links
        links = {
            re.sub(r'\s+', ' ', a.text.strip()): a['href']
            for a in soup.find_all('a', href=True)[:int(max_context / 1000)]
            if a.text.strip()
        }

        print(
            f"{colorama.Fore.CYAN}\n🕷️✅  Scrape successful!\n🧠🧠🧠analyzing the content... please wait...")
        emit('receive_message', {'status': 'info', 'message': f"🕷️✅  Scrape successful!"})
        emit('receive_message',
             {'status': 'info', 'message': "🧠🧠🧠analyzing the content... please wait..."})
        return {'text': text, 'links': links}
    except Exception as e:
        emit(f"{colorama.Fore.RED}\n🕷️❌  Scrape failed with error: {e}")
        print(f"Scraping failed with error: {e}")
        return f"Scraping failed with error: {e}"


class Agent:
    def __init__(self, base_url=None, api_key=None):
        self.tasks = {}
        self.global_history = []
        self.stop_processing = False
        self.task_stopped = False
        self.clear_global_history = False
        # Initialize client parameters with defaults
        client_params = {
            'base_url': base_url,
            'api_key': api_key or os.environ.get('OPENAI_API_KEY')
        }

        # Validate API key
        if not client_params['api_key']:
            raise ValueError(
                "API key not provided. Either pass it as 'api_key' or set it in the 'OPENAI_API_KEY' environment variable."
            )

        # Initialize OpenAI client
        try:
            self.client = OpenAI(**client_params)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

        # Retry settings
        self.max_retries = 3
        self.retry_delay = 1

        # Additional properties
        self.current_request = None
        self.tasks = {}
        self.global_history = []

    def generate_prompt(self, task: str, previous_actions: list) -> str:
        """
        Generates a dynamic prompt based on the task and previous actions.
        """
        complexity = "complex" if len(previous_actions) > 5 else "simple"

        if complexity == "complex":
            instructions = """
            For complex tasks, ensure intermediate results are validated before proceeding.
            Use structured approaches and avoid assumptions.
            """
        else:
            instructions = """
            For simple tasks, provide precise and concise responses to quickly achieve the goal.
            """

        action_definitions = """
        You are an AI with tools and actions.
        HE FORMAT FOR ACTIONS IS {ACTION} ARGUMENTS
        The following are the actions that fit the above format:
        1. {SEARCH} [QUERY] - Conduct a web search with a clear, focused query. Example: {SEARCH} weather in New York.
        You must Scrape between 2-6 results depending on task complexity.
        2. {SCRAPE} [URL] - 
        Only use {SCRAPE} if one or more of the following conditions are met: 
            a) You have the URL from search results
            b) You have the URL from a website you scraped
            c) The user included the URL in the task description. 
            In each case or cases you can only use the {SCRAPE} action on the URL provided.
        if you get a scrape error, Try to scrape another URL if you have one or search for a new URL.
        3. {DOWNLOAD} [URL] - Download a file from a URL. Example: {DOWNLOAD} https://example.com/file.pdf.
        4. {EXECUTE_PYTHON} [CODE] -  Run Python code. Example: {EXECUTE_PYTHON} print(42).
        5. {EXECUTE_BASH} [CODE] - Run a Bash command. Example: {EXECUTE_BASH} ls -l.
        6. {CONCLUDE} [CONCLUSION] - Provide a detailed summary once all tasks are completed. This should be used **only after all 
        actions have been executed** and the task is ready to conclude, 
        For research or scientific tasks, structure your conclusion as follows:
            {CONCLUDE}
            - Abstract – summary of the research objectives, methods, findings, and conclusions.
            - Introduction – Provide background, state the research problem, and outline objectives.
            - Literature Review – Summarize relevant studies and identify gaps.
            - Methodology – Describe the research design, sample size, and methods.
            - Results – Present findings (include tables/graphs if necessary).
            - Discussion – Interpret results, compare with existing studies, and discuss limitations.
            - Conclusion – Summarize findings and suggest future research.
            - References – List citations used.
        For all other cases just provide the summary like this: {CONCLUDE}: followed by the summary of the task.

        NEVER DO MORE THEN ONE ACTION IN A RESPONSE, NEVER DESCRIBE WHAT YOU ARE DOING, JUST DO
        """

        return f"""
        You are an AI assistant. Follow the rules:
        {action_definitions}
        {instructions}

        Task: {task}
        Previous Actions: {json.dumps(previous_actions or [])}
        Today's Date: {datetime.datetime.now().isoformat()}

        Remember: Do not use CONCLUDE until all necessary actions have been performed.
        """

    def get_conclusion(self, task, actions):
        messages = [
            {"role": "system", "content": "You are an AI agent that provides conclusions based on task completion."},
            {"role": "user",
             "content": f"Task: {task}\n\nActions taken: {json.dumps(actions)}\n\nProvide a conclusion for the task."}
        ]
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=base_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_completion_tokens=max_context,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.retry_delay * (attempt + 1))

    def stream_response(self, task: str, previous_actions: list) -> Generator[str, None, None]:
        """
        Streams the response from the AI, ensuring it follows the single-action rule and
        dynamically adapts to task complexity.
        """
        prompt = self.generate_prompt(task, previous_actions)
        messages = [{"role": "system", "content": prompt}]

        try:
            response = self.client.chat.completions.create(
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                model=base_model,
                messages=messages,
                max_tokens=max_tokens,
                max_completion_tokens=max_context,
                stream=True,
            )

            full_response = ""
            response_iterator = response.__iter__()

            while True:
                if self.stop_processing:
                    print("Stopping task processing...")
                    response.close()
                    self.task_stopped = True
                    break

                try:
                    # Get next chunk with timeout
                    chunk = next(response_iterator)
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        print(content, end='', flush=True)
                        yield content
                except StopIteration:
                    break

            if task in self.tasks:
                self.tasks[task]['streamed_response'] = full_response

        except Exception as e:
            error_type = type(e).__name__
            print(f"Error occurred: {error_type} - {str(e)}")
            emit('receive_message', {
                'status': 'error',
                'message': f"{error_type}: {str(e)}. Try rephrasing the task or checking input data."
            })

    def evaluate_completion(self, task, actions):
        if len(actions) > 2:
            messages = [
                {"role": "system", "content": "You are an AI agent that evaluates task completion."},
                {"role": "user",
                 "content": f"Task: {task}\n\nActions taken: {json.dumps(actions)}\n\nHas the task been completed? Respond with 'YES' if completed, 'NO' if not."}
            ]
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=base_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_completion_tokens=max_context,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty
                    )
                    return "YES" in response.choices[0].message.content.upper()
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise e
                    time.sleep(self.retry_delay * (attempt + 1))
            else:
                return

    def search_web(self, query):
        results = []
        seen_links = set()  # Track unique links to avoid duplicates

        # Base Google Search URL
        base_url = "https://www.google.com/search"
        params = {"q": query, "hl": "en"}  # Parameters for Google search query

        for page in range(2):  # Iterate through the first five pages
            params["start"] = page * 10  # Pagination parameter
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                print(f"Failed to fetch search results: {response.status_code}")
                break

            # Parse HTML content
            soup = BeautifulSoup(response.text, "html.parser")
            search_results = soup.select("div")

            for result in search_results:
                try:
                    title_element = result.select_one("h3")
                    link_element = result.select_one("a")

                    if title_element is not None and link_element is not None:
                        link = link_element["href"].replace('/url?q=', '')
                        if link not in seen_links:  # Check if the link is already seen
                            seen_links.add(link)  # Mark this link as seen
                            results.append({
                                "title": title_element.get_text(strip=True),
                                "link": link,
                            })
                except Exception as e:
                    print(f"Error parsing result: {e}")
                    continue

        return results

    def _scrape_python_files(self, path):
        """
        Collects the names and contents of all Python files in a folder and its subfolders,
        or from a single Python file if a file path is provided.

        Args:
            path (str): The file or folder path to scan.

        Returns:
            list[dict]: A list of dictionaries, each containing 'filename' and 'content' keys.
        """
        python_files_data = []

        if os.path.isfile(path):  # Check if the input is a file
            if path.endswith('.py'):  # Ensure it's a Python file
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    python_files_data.append({'filename': path, 'content': content})
                except Exception as e:
                    print(f"{colorama.Fore.RED}Error reading file {path}: {e}")
        elif os.path.isdir(path):  # Check if the input is a folder
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            python_files_data.append({'filename': file_path, 'content': content})
                        except Exception as e:
                            print(f"{colorama.Fore.RED}Error reading file {file_path}: {e}")
        else:
            print(f"{colorama.Fore.RED}Invalid path: {path} is neither a file nor a directory.")

        return python_files_data

    def _download_file(self, url: str, output_path=None) -> str:

        def has_video_content(url):
            try:
                # Get the webpage content
                response = requests.get(url, verify=False, headers=HEADERS)
                soup = BeautifulSoup(response.text, 'html.parser')

                # Check for common video elements
                video_elements = (
                        soup.find_all('video') or
                        soup.find_all('iframe', src=lambda x: x and ('youtube.com' in x or 'vimeo.com' in x)) or
                        'youtube.com' in url or
                        'vimeo.com' in url or
                        any(vid_site in url for vid_site in [
                            'dailymotion.com', 'twitter.com', 'tiktok.com',
                            'facebook.com', 'instagram.com', 'reddit.com'
                        ])
                )
                return bool(video_elements)
            except:
                return False

        # If video content is detected, use yt-dlp
        if has_video_content(url):
            try:
                output_path = output_path or os.getcwd()
                if not os.path.isdir(output_path):
                    output_path = os.path.dirname(output_path)

                ydl_opts = {
                    'format': 'bestvideo+bestaudio/best',  # Download best quality
                    'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
                    'quiet': True,
                    'no_warnings': True,
                    'progress_hooks': [
                        lambda d: print(
                            f"\rDownloading... {(d.get('downloaded_bytes', 0) / d.get('total_bytes', 1) * 100):.1f}%"
                            if d['status'] == 'downloading' and d.get('total_bytes')
                            else f"\rDownloading... {d.get('downloaded_bytes', 0) / 1024 / 1024:.1f}MB downloaded"
                            if d['status'] == 'downloading'
                            else "\nDownload completed. Processing...", end='')
                    ],
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_title = info['title']
                    video_path = os.path.join(output_path, f"{video_title}.{info['ext']}")
                    print(f"{colorama.Fore.GREEN}\n✅ Video downloaded successfully: {video_path}")
                    return f"✅ Video downloaded successfully: {video_path}"

            except Exception as e:
                print(f"{colorama.Fore.RED}\n❌ Failed to download video: {e}")
                return f"❌ Failed to download video: {e}"

        # If no video content or video download fails, do regular file download
        try:
            response = requests.get(url, stream=True, verify=False, headers=HEADERS)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            output_path = output_path or os.path.join(os.getcwd(), os.path.basename(url))
            if os.path.isdir(output_path):
                output_path = os.path.join(output_path, os.path.basename(url))
            with open(output_path, 'wb') as file, tqdm(
                    desc=f"Downloading {os.path.basename(output_path)}",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    progress_bar.update(size)
            print(f"{colorama.Fore.GREEN}\n✅ File downloaded successfully: {output_path}")
            return f"✅ File downloaded successfully: {output_path}"
        except requests.exceptions.RequestException as e:
            print(f"{colorama.Fore.RED}\n❌ Failed to download file: {e}")
            return f"❌ Failed to download file: {e}"
        except IOError as e:
            print(f"{colorama.Fore.RED}\n❌ Error saving file: {e}")
            return f"❌ Error saving file: {e}"

    def execute_code(self, code: str, language: str) -> Dict[str, Any]:
        temp_file = os.path.join(os.getcwd(), f"temp_{int(time.time())}.{'py' if language == 'python' else 'sh'}")
        try:
            if language == 'python':
                required_packages = extract_imports(code)
                if required_packages:
                    installation_results = install_missing_packages(required_packages)
                    if not all(installation_results.values()):
                        failed_packages = [pkg for pkg, success in installation_results.items() if not success]
                        return {'success': False, 'output': None,
                                'error': f"Failed to install required packages: {', '.join(failed_packages)}",
                                'return_code': -1}

            with open(temp_file, 'w') as f:
                f.write(code)
            result = subprocess.run([language, temp_file], capture_output=True, text=True, timeout=30)
            success = result.returncode == 0
            print(f"{'✅ Code executed successfully' if success else '❌ Code execution failed'}")
            print(f"Result: {result.stdout if success else result.stderr}")
            return {'success': success, 'output': result.stdout if success else result.stderr,
                    'error': result.stderr if not success else None, 'return_code': result.returncode}
        except Exception as e:
            print(f"💥 Code execution error: {e}")
            return {'success': False, 'output': None, 'error': str(
                e) + '\nMake sure you only send the command and the code, without anything else in your message',
                    'return_code': -1}
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def extract_actions(self, response: str) -> list:
        actions = []
        action_pattern = re.compile(r'\{([A-Z_]+)\}(.+?)(?=\{[A-Z_]+\}|$)', re.DOTALL)
        matches = action_pattern.findall(response)
        for action_type, action_content in matches:
            action = f"{{{action_type}}}{action_content.strip()}"
            if action not in actions:
                actions.append(action)
        return actions

    def execute_task(self, task):
        if task not in self.tasks:
            self.tasks[task] = {'previous_actions': [], 'conclusions': [], 'performed_actions': set()}

        task_context = self.tasks[task]
        if self.clear_global_history:
            self.global_history = []
            self.clear_global_history = False
        previous_actions = task_context['previous_actions'] + self.global_history
        conclusions = task_context['conclusions']
        performed_actions = task_context['performed_actions']
        max_steps = 20
        step = 0

        print(f"{colorama.Fore.CYAN}🚀 Starting task: {task}\n🧠🧠🧠Analyzing the task... please wait...")
        emit('receive_message', {'status': 'info', 'message': f"🚀 Starting task: {task}"})
        emit('receive_message', {'status': 'info', 'message': "🧠🧠🧠Analyzing the task... please wait..."})

        while step < max_steps and not self.task_stopped:
            step += 1

            full_response = ""
            for chunk in self.stream_response(task, previous_actions):
                full_response += chunk

            actions = self.extract_actions(full_response)

            for action in actions:

                performed_actions.add(action)

                if "{END_SESSION}" in action:
                    print("\n👋 Session ended by agent.")
                    emit('receive_message', {'status': 'info', 'message': "👋 Session ended by agent."})
                    step = max_steps
                    break

                elif "{CONCLUDE}" in action:
                    conclusion = action[11:].strip()
                    print(f"\n📊 Here's my conclusion:\n{conclusion}")
                    emit('receive_message', {'status': 'info', 'message': f"📊 Here's my conclusion:"})
                    emit('receive_message', {'status': 'info', 'message': conclusion})
                    conclusions.append(conclusion)
                    step = max_steps
                    break

                elif action.startswith("{SCRAPE_PYTHON}"):
                    python_project_files = action[16:].strip()
                    print(f"{colorama.Fore.CYAN}\nFinished scraping python files in {python_project_files}\n")
                    emit('receive_message',
                         {'status': 'info', 'message': f"Finished scraping python files in {python_project_files}"})
                    previous_actions.append(f"Scraped Python files in {python_project_files}")

                elif action.startswith("{SEARCH}"):
                    search_query = action[8:].strip().split('\n')[0].replace('"', '')
                    print(f"{colorama.Fore.CYAN}\n🔍 Searching web for: {search_query}")
                    emit('receive_message', {'status': 'info', 'message': f"🔍 Searching web for: {search_query}"})
                    search_result = self.search_web(search_query)
                    previous_actions.append(f"Searched: {search_query}")
                    previous_actions.append(f"Search results: {json.dumps(search_result)}\n Select between 2-4 results"
                                            f" to scrape or download")
                    print(f"{colorama.Fore.CYAN}\n🔍 Search results found: {json.dumps(len(search_result))}")
                    emit('receive_message',
                         {'status': 'info', 'message': f"🔍 Search results found: {json.dumps(len(search_result))}"})
                    emit('receive_message',
                         {'status': 'info', 'message': f"🧠🧠🧠 Analyzing the search results... please wait..."})

                elif action.startswith("{DOWNLOAD}"):
                    try:
                        url = re.search(r'{DOWNLOAD}\s*(https?://\S+)', action).group(1)
                        print(f"{colorama.Fore.CYAN}\n📥 Downloading file: {url}")
                        download_result = self._download_file(url)
                        previous_actions.append(f"Downloaded: {url} - {download_result}")
                        print(f"{colorama.Fore.CYAN}\n📥 Downloaded: {url} - {download_result}")
                        emit('receive_message',
                             {'status': 'info', 'message': f"📥 Downloaded: {url} - {download_result}"})
                    except ValueError as ve:
                        print(f"Value error: {ve}")
                    except AttributeError as ae:
                        print(f"Attribute error: {ae}")

                elif action.startswith("{SCRAPE}"):
                    match = re.search(r'{SCRAPE}\s*(https?://\S+)', action)
                    try:
                        url = match.group(1)
                        if url.endswith(".pdf"):
                            # Handle PDF scraping
                            pass
                        print(f"{colorama.Fore.CYAN}\n🕷️ Scraping website: {url}")
                        emit('receive_message', {'status': 'info', 'message': f"🕷️ Scraping website: {url}"})
                        result = scrape_website(url)
                        previous_actions.append(f"Scraped {url}")
                        previous_actions.append(f"Scraping results: {json.dumps(result)} is this the information you "
                                                f"were looking for?")
                    except Exception as e:
                        previous_actions.append(f"Scraping error: {str(e)}")
                        print(f"{colorama.Fore.RED}🕷️ Scraping error: {str(e)}")

                elif action.startswith("{EXECUTE_PYTHON}"):
                    code = action[16:].strip().removeprefix("```python").removesuffix("```").strip()
                    print(f"{colorama.Fore.CYAN}🐍 Executing Python code:\n```python\n{code}\n```")
                    emit('receive_message',
                         {'status': 'info', 'message': f"🐍 Executing Python code:\n```python\n{code}\n```"})
                    result = self.execute_code(code, 'python')
                    previous_actions.append(f"Executed Python: {code}")
                    previous_actions.append(f"Result: {result}")
                    print(f"{colorama.Fore.CYAN}🐍 Result:\n```markdown\n{result}\n```")
                    emit('receive_message', {'status': 'info', 'message': f"🐍 Result:\n```markdown\n{result}\n```"})

                elif action.startswith("{EXECUTE_BASH}"):
                    code = action[14:].strip().removeprefix("```bash").removesuffix("```").strip()
                    print(f"{colorama.Fore.CYAN}💻 Executing Bash code:\n{code}")
                    emit('receive_message', {'status': 'info', 'message': f"💻 Executing Bash code:\n{code}"})
                    result = self.execute_code(code, 'bash')
                    previous_actions.append(f"Executed Bash: {code}")
                    previous_actions.append(f"Result: {result}")
                    print(f"{colorama.Fore.CYAN}💻 Result: {result}")
                    emit('receive_message', {'status': 'info', 'message': f"💻 Result: {result}"})

            self.tasks[task] = {'previous_actions': previous_actions, 'conclusions': conclusions,
                                'performed_actions': performed_actions}

            if self.evaluate_completion(task, previous_actions):
                print("🎉 Task completed successfully!\nWorking on creating a conclusion...🧠🧠🧠")
                emit('receive_message', {'status': 'info', 'message': "🎉 Task completed successfully!"})
                emit('receive_message', {'status': 'info', 'message': "Working on creating a conclusion...🧠🧠🧠"})
                break

        if not conclusions:
            conclusion = self.get_conclusion(task, previous_actions)
            if conclusion:
                conclusions.append(conclusion)
                previous_actions.append(f"Added conclusion: {conclusion}")

        if conclusions:
            print("\n📊 Conclusions:\n")
            for conclusion in conclusions:
                print(conclusion)
                emit('receive_message', {'status': 'info', 'message': conclusion})

        emit('hide_waiting_animation')
        return len(conclusions) > 0



def main():
    agent = Agent(base_url=base_url, api_key=base_api)
    current_task = ""
    while True:
        if current_task:
            print(f"\n{colorama.Fore.CYAN}🔄 Current task: {current_task}\nEnter your task (or 'quit' to exit):")
        else:
            print("\nEnter your task (or 'quit' to exit):")
        task_input = input("INPUT: ").strip()

        if task_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not task_input:
            print("Please enter a valid task.")
            continue

        task = task_input
        current_task = task_input

        try:
            print("\n" + "=" * 50)
            agent.execute_task(task)
            print("=" * 50)
        except KeyboardInterrupt:
            print("\n🛑 Task interrupted by user.")
            continue
        except Exception as e:
            print(f"\n{colorama.Fore.RED}❌ Error executing task: {e}")
            continue


if __name__ == "__main__":
    main()
