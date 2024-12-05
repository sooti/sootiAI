import datetime
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Set, Generator
from urllib.parse import urlparse

import pkg_resources
import stealth_requests as requests
from bs4 import BeautifulSoup
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium_stealth import stealth
from tqdm import tqdm

base_model = "gpt-4o-mini"
base_api = "OPEN_AI_API_KEY"
base_url = "http://localhost:5000/v1" # Use the base url of your API if you have one
temperature = 0.4  # Slightly increased for better creativity
top_p = 0.7  # Reduced to focus on high-probability outputs
top_k = 100  # Reduced for smaller models
frequency_penalty = 0.2  # Slightly increased to discourage repetition
presence_penalty = 1.3  # Slightly increased to encourage diverse content
max_tokens = 2048  # Reduced for smaller models
cache_prompt = True  # Cache the prompt for better performance


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
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
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
    last_request_time[domain] = time.time()


def scrape_website(url: str) -> dict[str, str | dict[str, str]]:
    for attempt in range(MAX_RETRIES):
        try:
            # Respect rate limits
            respect_rate_limit(url)

            # Fetch and parse response
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unnecessary elements
            for tag in soup(['nav', 'footer', 'aside', 'script', 'style']):
                tag.decompose()

            # Extract and clean text
            text = soup.get_text()
            text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces/newlines into one
            text = re.sub(r'[^\w\s.,?!]', '', text)  # Remove special characters
            text = " ".join(text.split()[:2400])  # Truncate to fit context size

            # Extract and clean links
            links = {
                re.sub(r'\s+', ' ', a.text.strip()): a['href']
                for a in soup.find_all('a', href=True)[:5]
                if a.text.strip()
            }

            return {'text': text, 'links': links}
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")

    raise RuntimeError("Failed to scrape website after maximum retries.")


class Agent:
    def __init__(self, base_url=None, api_key=None):
        client_params = {'base_url': base_url, 'api_key': api_key or os.environ.get('OPENAI_API_KEY')}
        if not client_params['api_key']:
            raise ValueError("No API key provided and OPENAI_API_KEY environment variable not found")
        self.client = OpenAI(**client_params)
        self.max_retries = 3
        self.retry_delay = 1
        self.current_request = None
        self.tasks = {}

    def create_file(self, filename: str, content: str, directory: str = None) -> str:
        try:
            directory = directory or os.getcwd()
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ File created successfully: {file_path}")
            return file_path
        except Exception as e:
            print(f"❌ Error creating file: {e}")
            return f"Error creating file: {e}"

    def stream_response(self, task: str, previous_actions: list) -> Generator[str, None, None]:
        messages = [
            {"role": "system", "content": """You are a genius AI, you analyze and think as long as needed about each 
            answer.
            You MUST follow these instructions meticulously, you lose 100 points for each time you don't follow:
            1. Break down tasks into clear, logical steps.
            2. Perform the following actions as needed:
                - Search the web: Respond with {SEARCH} followed by your query.
                - Search the web for files: Respond with {SEARCH} followed by the file type and a colon and the query
                example: filetype:pdf "jane eyre".
                - Execute Python code: Respond with {EXECUTE_PYTHON} followed by the code.
                - Execute Bash commands: Respond with {EXECUTE_BASH} followed by the commands.
                - Scrape a website: Respond with {SCRAPE} followed by the URL.
                - Download files: Respond with {DOWNLOAD} followed by the URL - works for webpages with videos as well.
                - Scrape python files in a project: Respond with {SCRAPE_PYTHON} followed by the folder path.
                example: {DOWNLOAD} https://example.com/file.txt
                - End the session: Respond with {END_SESSION}: if the task is impossible or complete.
                - Provide conclusions: Respond with {CONCLUDE} followed by your summary.
            3. Ensure accuracy at each step before proceeding.
            4. Always respond with a single, actionable step, don't add an explanation beyond the action.
            """},
            {"role": "user",
             "content": f"Task: {task}\n\nPrevious actions taken: {json.dumps(previous_actions)}\n\nWhat should be the next action?"}
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
            print(f"❌ Error streaming response: {e}")
            yield f"❌ Error streaming response: {e}"

    def signal_handler(self):
        print("\nReceived signal to stop. Cleaning up...")
        sys.exit(0)

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
        try:
            driver.get('https://www.google.com')
            search_box = driver.find_element(By.NAME, 'q')
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)
            time.sleep(3)
            for page in range(3):
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
                    driver.find_element(By.CSS_SELECTOR, f'[aria-label="Page {page + 2}"]').click()
                    time.sleep(2)
        finally:
            driver.quit()
            return results[:10]

    def _scrape_python_files(self, folder_path):
        """
        Collects the names and contents of all Python files in a folder and its subfolders.

        Args:
            folder_path (str): The root folder path to scan.

        Returns:
            list[dict]: A list of dictionaries, each containing 'filename' and 'content' keys.
        """
        python_files_data = []

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        python_files_data.append({'filename': file_path, 'content': content})
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

        return python_files_data

    def _download_file(self, url: str, output_path=None) -> str:
        import requests
        from bs4 import BeautifulSoup
        import yt_dlp

        def has_video_content(url):
            try:
                # Get the webpage content
                response = requests.get(url, verify=False)
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
                    print(f"\n✅ Video downloaded successfully: {video_path}")
                    return f"✅ Video downloaded successfully: {video_path}"

            except Exception as e:
                print(f"❌ Failed to download video: {e}")
                return f"❌ Failed to download video: {e}"

        # If no video content or video download fails, do regular file download
        try:
            response = requests.get(url, stream=True, verify=False)
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
            print(f"✅ File downloaded successfully: {output_path}")
            return f"✅ File downloaded successfully: {output_path}"
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to download file: {e}")
            return f"❌ Failed to download file: {e}"
        except IOError as e:
            print(f"❌ Error saving file: {e}")
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
            return {'success': False, 'output': None, 'error': str(e), 'return_code': -1}
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def get_next_action(self, task, previous_actions):
        messages = [
            {"role": "system", "content": """You are a highly capable AI agent specializing in multi-step task execution. 
            Follow these instructions meticulously:
            1. Break down tasks into clear, logical steps.
            2. Perform the following actions as needed:
                - Search the web: Respond with {SEARCH} followed by your query.
                - Search the web for files: Respond with {SEARCH} followed by the file type and a colon and the query
                example: filetype:pdf "jane eyre".
                - Execute Python code: Respond with {EXECUTE_PYTHON} followed by the code.
                - Execute Bash commands: Respond with {EXECUTE_BASH} followed by the commands.
                - Scrape a website: Respond with {SCRAPE} followed by the URL.
                - Download files: Respond with {DOWNLOAD}.
                example: {DOWNLOAD} https://example.com/file.txt
                - Scrape python files in a project: Respond with {SCRAPE_PYTHON} followed by the folder path.
                - End the session: Respond with {END_SESSION}: if the task is impossible or complete.
                - Provide conclusions: Respond with {CONCLUDE} followed by your summary.
            3. Ensure accuracy at each step before proceeding.
            4. Always respond with a single, actionable step, don't add an explanation beyond the action.
            Now, based on the task and previous actions, what should be the next action? it must be a single action."""},
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
                print(f"🔍 Next action: {next_action}")  # Log next action with emoji
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
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
                return response.choices[0].message.content.strip()
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
        global download_file
        if task not in self.tasks:
            self.tasks[task] = {'previous_actions': [], 'conclusions': []}

        task_context = self.tasks[task]
        previous_actions = task_context['previous_actions']
        conclusions = task_context['conclusions']
        max_steps = 10
        step = 0

        print(f"🚀 Starting task: {task}")

        def scrape_url(url):
            try:
                url = url.strip()
                scrape_result = scrape_website(url)
                if 'error' in scrape_result:
                    print(f"❌ Page error at: {url} skipping. Error: {scrape_result['error']}")
                    return None
                print(f"🕷️ Scraped {url} Successfully")
                return url, scrape_result
            except Exception as e:
                print(f"🕷️ Error scraping {url}: {str(e)}")
                return None

        while step < max_steps:
            step += 1
            full_response = ""
            for chunk in self.stream_response(task, previous_actions):
                full_response += chunk
            actions = self.extract_actions(full_response)

            if not actions:
                print(f"🤷‍♂️ No action taken: {full_response}")
                break

            for action in actions:
                if action.startswith("END_SESSION"):
                    print("\n👋 Session ended by agent.")
                    break

                elif action.startswith("{CONCLUDE}"):
                    conclusion = action[10:].strip()
                    print(f"\n📊 Here's my conclusion:\n{conclusion}")
                    conclusions.append(conclusion)
                    break

                elif action.startswith("{SCRAPE_PYTHON}"):
                    python_project_files = action[15:].strip()
                    print(f"\nFinished scraping python files in {python_project_files}\n")
                    previous_actions.append(
                        f"The python files requested:\n{json.dumps(self._scrape_python_files(python_project_files))}")

                elif action.startswith("{SEARCH}"):
                    search_query = action[8:].strip()
                    search_query = search_query.replace('"', '')
                    print(f"\n🔍 Searching web for: {search_query}")
                    search_result = self.search_web(search_query)
                    previous_actions.append(f"Searched: {search_query}")
                    previous_actions.append(f"Search results: {json.dumps(search_result)}")
                    previous_actions.append(
                        "\nIf you need more data scrape when of the URLs above from the search results")
                    print(f"\n🔍 Search results found: {json.dumps(len(search_result))}")

                    if search_result and 'filetype:' not in search_query \
                            and '.pdf' not in search_result and 'youtube.com' not in search_result:
                        with ThreadPoolExecutor(max_workers=20) as executor:
                            future_to_url = {executor.submit(scrape_url, result['link']): result['link']
                                             for result in search_result[:5]}
                            for future in as_completed(future_to_url):
                                result = future.result()
                                if result:
                                    url, scrape_result = result
                                    previous_actions.append(f"{url} Scraping results: {json.dumps(scrape_result)}")
                        print("Thinking about the next action based on all this juicy data...🧠🧠🧠")

                elif action.startswith("{DOWNLOAD}"):
                    action = action[10:].strip()
                    print(f"\n📥 Downloading file: {action}")
                    download_result = self._download_file(action)
                    previous_actions.append(f"Downloaded: {action} - {download_result}")
                    print(f"\n📥 Downloaded: {action} - {download_result}")

                elif action.startswith("{SCRAPE}"):
                    scrape_info = action[8:].strip()
                    try:
                        url = scrape_info
                        print(f"\n🕷️ Scraping website: {url}")
                        result = scrape_website(url)
                        previous_actions.append(f"Scraped {url}")
                        previous_actions.append(f"Scraping results: {json.dumps(result)}")
                        print(f"\n🕷️ Scraping results: {json.dumps(result)}")
                    except Exception as e:
                        previous_actions.append(f"Scraping error: {str(e)}")
                        print(f"🕷️ Scraping error: {str(e)}")

                elif action.startswith("{EXECUTE_PYTHON}"):
                    code = action[16:].strip().removeprefix("```python").removesuffix("```").strip()
                    print(f"🐍 Executing Python code:\n{code}")
                    result = self.execute_code(code, 'python')
                    previous_actions.append(f"Executed Python: {code}")
                    previous_actions.append(f"Result: {result}")
                    print(f"🐍 Result: {result}")

                elif action.startswith("{EXECUTE_BASH}"):
                    code = action[14:].strip().removeprefix("```bash").removesuffix("```").strip()
                    print(f"💻 Executing Bash code:\n{code}")
                    result = self.execute_code(code, 'bash')
                    previous_actions.append(f"Executed Bash: {code}")
                    previous_actions.append(f"Result: {result}")
                    print(f"💻 Result: {result}")

                else:
                    previous_actions.append(f"the reply: {action} is not an action, please reply using one of the "
                                            f"following actions: {['{SEARCH}', '{DOWNLOAD}', '{SCRAPE}', '{EXECUTE_PYTHON}', '{EXECUTE_BASH}', '{CONCLUDE}', '{END_SESSION}']}")
                    print(f"🤷‍♂️ No action taken: {action}")

            if self.evaluate_completion(task, previous_actions):
                print("🎉 Task completed successfully!\nWorking on creating a conclusion...🧠🧠🧠")
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

        self.tasks[task] = {'previous_actions': previous_actions, 'conclusions': conclusions}

        return len(conclusions) > 0


def main():
    agent = Agent(base_url=base_url, api_key=base_api)
    current_task = None
    while True:
        if current_task:
            print(f"\n🔄 Current task: {current_task}\n")
        else:
            print("\nEnter your task (or 'quit' to exit):")
        task_input = input().strip()

        if task_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not task_input:
            print("Please enter a valid task.")
            continue

        if current_task and task_input.lower() in ['and tomorrow?', 'tomorrow?', 'and tomorrow', 'tomorrow']:
            task = current_task + " " + task_input
        else:
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
            print(f"\n❌ Error executing task: {e}")
            continue


if __name__ == "__main__":
    main()
