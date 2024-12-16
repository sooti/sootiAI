import datetime
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, Set, Generator
from urllib.parse import urlparse
import yt_dlp

import pkg_resources
import requests
import urllib3
from bs4 import BeautifulSoup
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
from tqdm import tqdm
import colorama

colorama.init(autoreset=True)
urllib3.disable_warnings()

base_model = "gpt-4o"
base_api = "OPEN_API_KEY"  # Use your OpenAI API key
base_url = "http://localhost:5000/v1"  # Use the base url of your API if you have one
temperature = 0.3
top_p = 0.7
frequency_penalty = 0.2
presence_penalty = 1.6
max_tokens = 2048  # This value is fine
max_context = 32000

# Directory to store session data
DATA_DIR = "data"


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
        response = requests.get(url, verify=False, headers=HEADERS)
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

        return {'text': text, 'links': links}
    except Exception as e:
        print(f"Scraping failed with error: {e}")
        return f"Scraping failed with error: {e}"


class Agent:
    def __init__(self, base_url=None, api_key=None):
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

    def stream_response(self, task: str, previous_actions: list) -> Generator[str, None, None]:
        messages = [
            {"role": "system", "content": """You are a genius AI, you analyze and think as long as needed about each 
            answer, don't EVER explain the step, until you reach the conclusion phase, just run the action!
            NEVER DO MULTIPLE DIFFERENT ACTIONS IN ONE STEP, EXAMPLE: {SCRAPE} https://exmaple.com {DOWNLOAD} https://cnn.com/file
            , ALWAYS DO ONE ACTION AT A TIME.
            You MUST follow these instructions meticulously, you lose 100 points for each time you don't follow:
            1. Break down tasks into clear, logical steps.
            2. Perform the following actions as needed:
                - Web search: Respond with {SEARCH} followed by your query, try to create a logical search query 
                that will yield the most effective results, don't use long queries, if you need to look for multiple
                items, example, nvidia stock, intel stock, just do one search at a time for each.
                - File web search: Respond with {SEARCH} followed by the file type and a colon and the query
                example of a websearch of files: filetype:pdf "jane eyre".
                - Execute Python code: Respond with {EXECUTE_PYTHON} followed by the code, never use anything with APIs 
                that require signup.
                - Execute Bash commands: Respond with {EXECUTE_BASH} followed by the commands.
                - Scrape a website: Respond with {SCRAPE} followed by the URL.
                - Download files: Respond with {DOWNLOAD} followed by the URL - works for webpages with videos as well.
                - Scrape python files in a project: Respond with {SCRAPE_PYTHON} followed by the folder path.
                example: {DOWNLOAD} https://example.com/file.txt
                - End the session: Respond with {END_SESSION} if the task is impossible or complete.
                - Provide conclusions: Respond with {CONCLUDE} followed by your summary, do this ONLY if ALL of your
                tasks are done and you are ready to provide the summary and end the session, never do it in the same
                step as another action, it should be its own action.
                The conclusion should be in a format similar to this if its concluding research or information gathering:
                Abstract – 200-300 words summarizing the research objectives, methods, findings, and conclusions.
                Introduction – Provide background, state the research problem, and outline objectives.
                Literature Review – Summarize relevant studies and identify gaps.
                Methodology – Describe the research design, sample size, and methods.
                Results – Present findings (include tables/graphs if necessary).
                Discussion – Interpret results, compare with existing studies, and discuss limitations.
                Conclusion – Summarize findings and suggest future research.
                References – List citations used.
            3. Ensure accuracy at each step before proceeding.
            4. Always respond with a single, actionable step, don't add an explanation beyond the action.
            """},
            {"role": "user",
             "content": f"Task: {task}\n\nPrevious actions taken: {json.dumps(previous_actions)}\n\nFor context Today's date is "
                        f"{datetime.datetime.now()} What should be the next action?"}
        ]

        try:
            response = self.client.chat.completions.create(
                model=base_model,
                messages=messages,
                max_tokens=max_tokens,
                stream=True
            )
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(content, end='', flush=True)  # Log streaming response
                    yield content
            if task in self.tasks:
                self.tasks[task]['streamed_response'] = full_response
        except Exception as e:
            print(" ")

    def search_web(self, query):
        results = []
        options = Options()
        options.add_argument("--headless")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        driver = webdriver.Chrome(options=options)
        stealth(driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
                )
        driver.get('https://www.google.com')
        search_box = driver.find_element(By.NAME, 'q')
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2)
        for page in range(2):
            for result in driver.find_elements(By.CSS_SELECTOR, 'div.g'):
                try:
                    results.append({
                        'title': result.find_element(By.CSS_SELECTOR, "h3").text,
                        'link': result.find_element(By.CSS_SELECTOR, "a").get_attribute("href"),
                        'summary': result.find_element(By.CSS_SELECTOR, ".VwiC3b").text
                    })
                except:
                    pass
            if page == 0:
                try:
                    driver.find_element(By.CSS_SELECTOR, f'[aria-label="Page {page + 2}"]').click()
                except Exception:
                    continue
                time.sleep(2)

        return results[:int(max_context / 500)]

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
            return {'success': False, 'output': None, 'error': str(e) + '\nMake sure you only send the command and the code, without anything else in your message', 'return_code': -1}
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def get_next_action(self, task, previous_actions):
        messages = [
            {"role": "system", "content": """You are a genius AI, you analyze and think as long as needed about each 
            answer, don't EVER explain the step, until you reach the conclusion phase, just run the action!
            NEVER DO MULTIPLE DIFFERENT ACTIONS IN ONE STEP, EXAMPLE: {SCRAPE} https://exmaple.com {DOWNLOAD} https://cnn.com/file
            , ALWAYS DO ONE ACTION AT A TIME.
            You MUST follow these instructions meticulously, you lose 100 points for each time you don't follow:
            1. Break down tasks into clear, logical steps.
            2. Perform the following actions as needed:
                - Web search: Respond with {SEARCH} followed by your query, try to create a logical search query 
                that will yield the most effective results, don't use long queries, if you need to look for multiple
                items, example, nvidia stock, intel stock, just do one search at a time for each.
                - File web search: Respond with {SEARCH} followed by the file type and a colon and the query
                example: filetype:pdf "jane eyre".
                - Execute Python code: Respond with {EXECUTE_PYTHON} followed by the code.
                - Execute Bash commands: Respond with {EXECUTE_BASH} followed by the commands.
                - Scrape a website: Respond with {SCRAPE} followed by the URL.
                - Download files: Respond with {DOWNLOAD} followed by the URL - works for webpages with videos as well.
                - Scrape python files in a project: Respond with {SCRAPE_PYTHON} followed by the folder path.
                example: {DOWNLOAD} https://example.com/file.txt
                - End the session: Respond with {END_SESSION} if the task is impossible or complete.
                - Provide conclusions: Respond with {CONCLUDE} followed by your summary, do this ONLY if ALL of your
                tasks are done and you are ready to provide the summary and end the session, never do it in the same
                step as another action, it should be its own action.
                The conclusion should be simple for simple actions, in the case of research actions it must be
                 in a format similar to this if its concluding research or information gathering:
                Abstract – 200-300 words summarizing the research objectives, methods, findings, and conclusions.
                Introduction – Provide background, state the research problem, and outline objectives.
                Literature Review – Summarize relevant studies and identify gaps.
                Methodology – Describe the research design, sample size, and methods.
                Results – Present findings (include tables/graphs if necessary).
                Discussion – Interpret results, compare with existing studies, and discuss limitations.
                Conclusion – Summarize findings and suggest future research.
                References – List citations used.
            3. Ensure accuracy at each step before proceeding.
            4. Always respond with a single, actionable step, don't add an explanation beyond the action."""},
            {"role": "user",
             "content": f"Todays date is: {datetime.datetime.now()} your Task: {task}\n\nPrevious actions taken: {json.dumps(previous_actions)}\n\nWhat should be the next action?"}
        ]
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=base_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                next_action = response.choices[0].message.content
                print(f"{colorama.Fore.CYAN}🔍 Next action: {next_action}")  # Log next action with emoji
                return next_action
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.retry_delay * (attempt + 1))

    def evaluate_completion(self, task, actions):
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
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
                return "YES" in response.choices[0].message.content.upper()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.retry_delay * (attempt + 1))

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
            self.tasks[task] = {'previous_actions': [], 'conclusions': []}

        task_context = self.tasks[task]
        previous_actions = task_context['previous_actions'] + self.global_history
        conclusions = task_context['conclusions']
        max_steps = 50
        step = 0

        print(f"{colorama.Fore.CYAN}🚀 Starting task: {task}\n🧠🧠🧠Analyzing the task... please wait...")

        while step < max_steps:

            full_response = ""
            for chunk in self.stream_response(task, previous_actions):
                full_response += chunk
            actions = self.extract_actions(full_response)

            if not actions:
                print(f"{colorama.Fore.YELLOW}🤷‍♂️ No action taken: {full_response}\n")
                previous_actions.append(f"the reply: {full_response} is not an action, you MUST reply using one of the "
                                        f"following actions: {['{SEARCH}', '{DOWNLOAD}', '{SCRAPE}', '{EXECUTE_PYTHON}', '{EXECUTE_BASH}', '{CONCLUDE}', '{END_SESSION}']}")

            for action in actions:
                if action.startswith("{END_SESSION}"):
                    print("\n👋 Session ended by agent.")
                    step = max_steps
                    break

                elif "{CONCLUDE}" in action or "summary" in action or "conclusion" in action:
                    conclusion = action[10:].strip()
                    print(f"\n📊 Here's my conclusion:\n{conclusion}")
                    conclusions.append(conclusion)
                    step = max_steps
                    break

                elif action.startswith("{SCRAPE_PYTHON}"):
                    python_project_files = action[15:].strip()
                    print(f"{colorama.Fore.CYAN}\nFinished scraping python files in {python_project_files}\n")
                    previous_actions.append(
                        f"The python files requested:\n{json.dumps(self._scrape_python_files(python_project_files))}")

                elif action.startswith("{SEARCH}"):
                    search_query = action[8:].strip()
                    search_query = search_query.replace('"', '')
                    print(f"{colorama.Fore.CYAN}\n🔍 Searching web for: {search_query}")
                    search_result = self.search_web(search_query)
                    previous_actions.append(f"Searched: {search_query}")
                    previous_actions.append(f"Search results: {json.dumps(search_result)}")
                    previous_actions.append(
                        "\nYou must select at least between 2 and 8 results from the search results to scrape using the {SCRAPE} "
                        "action"
                        "select the ones that you think will yield the most useful information."
                        "If the information returned is not sufficient, try again and scrape one of the other results."
                        "If the information is sufficient, you can proceed to the next action.")
                    print(f"{colorama.Fore.CYAN}\n🔍 Search results found: {json.dumps(len(search_result))}")
                    break

                elif action.startswith("{DOWNLOAD}"):
                    try:
                        match = re.search(r'{DOWNLOAD}\s*(https?://\S+)', action)
                        if match is None:
                            raise ValueError("No match found for the provided action.")
                        url = match.group(1)
                        print(f"Extracted URL: {url}")
                    except ValueError as ve:
                        print(f"Value error: {ve}")
                    except AttributeError as ae:
                        print(f"Attribute error: {ae}")
                    print(f"{colorama.Fore.CYAN}\n📥 Downloading file: {url}")
                    download_result = self._download_file(url)
                    previous_actions.append(f"Downloaded: {url} - {download_result}")
                    print(f"{colorama.Fore.CYAN}\n📥 Downloaded: {url} - {download_result}")
                    break

                elif action.startswith("{SCRAPE}"):
                    match = re.search(r'{SCRAPE}\s*(https?://\S+)', action)
                    try:
                        url = match.group(1)
                        if url.endswith(".pdf") \
                                or url.endswith(".doc") \
                                or url.endswith(".docx") \
                                or url.endswith(".mp4"):
                            previous_actions.append("Please use {DOWNLOAD} action for downloading files.")
                            break
                        print(f"{colorama.Fore.CYAN}\n🕷️ Scraping website: {url}")
                        result = scrape_website(url)
                        previous_actions.append(f"Scraped {url}")
                        previous_actions.append(f"Scraping results: {json.dumps(result)} is this the information you "
                                                f"were looking for?"
                                                f"is it sufficient? if not, select an "
                                                f"additional website to scrape.")
                        print(f"{colorama.Fore.CYAN}\n🕷️✅  Scrape successful!\n🧠🧠🧠analyzing the content... please wait...")
                    except Exception as e:
                        previous_actions.append(f"Scraping error: {str(e)}")
                        print(f"{colorama.Fore.RED}🕷️ Scraping error: {str(e)}")

                elif action.startswith("{EXECUTE_PYTHON}"):
                    code = action[16:].strip().removeprefix("```python").removesuffix("```").strip()
                    print(f"{colorama.Fore.CYAN}🐍 Executing Python code:\n{code}")
                    result = self.execute_code(code, 'python')
                    previous_actions.append(f"Executed Python: {code}")
                    previous_actions.append(f"Result: {result}")
                    print(f"{colorama.Fore.CYAN}🐍 Result: {result}")
                    break

                elif action.startswith("{EXECUTE_BASH}"):
                    code = action[14:].strip().removeprefix("```bash").removesuffix("```").strip()
                    print(f"{colorama.Fore.CYAN}💻 Executing Bash code:\n{code}")
                    result = self.execute_code(code, 'bash')
                    previous_actions.append(f"Executed Bash: {code}")
                    previous_actions.append(f"Result: {result}")
                    print(f"{colorama.Fore.CYAN}💻 Result: {result}")
                    break

                elif action.startswith("{CONCLUDE}"):
                    conclusion = action[10:].strip()
                    print(f"{colorama.Fore.CYAN}\n📊 Here's my conclusion:\n\n\n===============================\n"
                          f"{conclusion}===============================")
                    conclusions.append(conclusion)
                    break

                else:
                    previous_actions.append(f"the reply: {action} is not an action, please reply using one of the "
                                            f"following actions: {['{SEARCH}', '{DOWNLOAD}', '{SCRAPE}', '{EXECUTE_PYTHON}', '{EXECUTE_BASH}', '{CONCLUDE}', '{END_SESSION}']}")
                    print(f"{colorama.Fore.YELLOW}🤷‍♂️ No action taken: {action}")
                    break

        self.tasks[task] = {'previous_actions': previous_actions, 'conclusions': conclusions}
        self.global_history.extend(previous_actions)

        return len(conclusions) > 0

    def save_session(self):
        if self.session_file:
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(self.tasks, f, indent=4)
            print(f"{colorama.Fore.GREEN}✅ Session saved to {self.session_file}")

    def load_session(self, session_file):
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                self.tasks = json.load(f)
            self.session_file = session_file
            print(f"{colorama.Fore.GREEN}✅ Session loaded from {session_file}")
        except FileNotFoundError:
            print(f"{colorama.Fore.RED}❌ Session file not found: {session_file}")


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
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
