
class CachedCodeAgent(CodeAgent):
    """
    CodeAgent with agentic plan caching for GAIA tasks.

    Reuses plan templates from previous runs by extracting keywords from tasks
    and adapting cached plan templates to new tasks.

    Args:
        plan_cache_size (int): Max size of cache, default 100.
        similarity_threshold (float): Threshold for semantic similarity matching, default 0.6.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, plan_cache_size: int = 100, similarity_threshold: float = 0.6, small_model: Model = None, **kwargs):
        self.plan_cache_size = plan_cache_size
        self.similarity_threshold = similarity_threshold
        self.cache = GAIAPlanCache(plan_cache_size)
        
        self.small_model_tokens = TokenUsage(input_tokens=0, output_tokens=0)
        self.large_model_tokens = TokenUsage(input_tokens=0, output_tokens=0)
        
        # Store small model for execution
        self.small_model = small_model
        
        # Initialize OpenAI client for plan adaptation
        self.client = OpenAI()
        
        super().__init__(**kwargs)
    
    def _track_small_model_tokens(self, input_tokens: int, output_tokens: int):
        """Track tokens used by small model calls."""
        self.small_model_tokens.input_tokens += input_tokens
        self.small_model_tokens.output_tokens += output_tokens
    
    def _track_large_model_tokens(self, input_tokens: int, output_tokens: int):
        """Track tokens used by large model calls."""
        self.large_model_tokens.input_tokens += input_tokens
        self.large_model_tokens.output_tokens += output_tokens
    
    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        """
        Override step stream to use small model for execution and track tokens properly.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()
        ### Generate model output ###
        memory_step.model_input_messages = input_messages
        stop_sequences = ["Observation:", "Calling tools:"]
        if self.code_block_tags[1] not in self.code_block_tags[0]:
            # If the closing tag is contained in the opening tag, adding it as a stop sequence would cut short any code generation
            stop_sequences.append(self.code_block_tags[1])
        try:
            additional_args: dict[str, Any] = {}
            if self._use_structured_outputs_internally:
                additional_args["response_format"] = CODEAGENT_RESPONSE_FORMAT
            
            # Use small model for execution if available, otherwise fall back to main model
            execution_model = self.small_model if self.small_model else self.model
            
            if self.stream_outputs:
                output_stream = execution_model.generate_stream(
                    input_messages,
                    stop_sequences=stop_sequences,
                    **additional_args,
                )
                chat_message_stream_deltas: list[ChatMessageStreamDelta] = []
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        chat_message_stream_deltas.append(event)
                        live.update(
                            Markdown(agglomerate_stream_deltas(chat_message_stream_deltas).render_as_markdown())
                        )
                        yield event
                chat_message = agglomerate_stream_deltas(chat_message_stream_deltas)
                memory_step.model_output_message = chat_message
                output_text = chat_message.content
            else:
                chat_message: ChatMessage = execution_model.generate(
                    input_messages,
                    stop_sequences=stop_sequences,
                    **additional_args,
                )
                memory_step.model_output_message = chat_message
                output_text = chat_message.content
                self.logger.log_markdown(
                    content=output_text,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            if not self._use_structured_outputs_internally:
                # This adds the end code sequence (i.e. the closing code block tag) to the history.
                # This will nudge subsequent LLM calls to finish with this end code sequence, thus efficiently stopping generation.
                if output_text and not output_text.strip().endswith(self.code_block_tags[1]):
                    output_text += self.code_block_tags[1]
                    memory_step.model_output_message.content = output_text

            memory_step.token_usage = chat_message.token_usage
            memory_step.model_output = output_text
            
            # Track tokens for the execution model (small model)
            if chat_message.token_usage:
                if execution_model == self.small_model:
                    self._track_small_model_tokens(
                        chat_message.token_usage.input_tokens,
                        chat_message.token_usage.output_tokens
                    )
                else:
                    self._track_large_model_tokens(
                        chat_message.token_usage.input_tokens,
                        chat_message.token_usage.output_tokens
                    )
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        ### Parse output ###
        try:
            if self._use_structured_outputs_internally:
                code_action = json.loads(output_text)["code"]
                code_action = extract_code_from_text(code_action, self.code_block_tags) or code_action
            else:
                code_action = parse_code_blobs(output_text, self.code_block_tags)
            code_action = fix_final_answer_code(code_action)
            memory_step.code_action = code_action
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)

        tool_call = ToolCall(
            name="python_interpreter",
            arguments=code_action,
            id=f"call_{len(self.memory.steps)}",
        )
        yield tool_call
        memory_step.tool_calls = [tool_call]

        ### Execute action ###
        self.logger.log_code(title="Executing parsed code:", content=code_action, level=LogLevel.INFO)
        try:
            code_output = self.python_executor(code_action)
            execution_outputs_console = []
            if len(code_output.logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(code_output.logs),
                ]
            observation = "Execution logs:\n" + code_output.logs
        except Exception as e:
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)

        truncated_output = truncate_content(str(code_output.output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        if not code_output.is_final_answer:
            execution_outputs_console += [
                Text(
                    f"Out: {truncated_output}",
                ),
            ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = code_output.output
        yield ActionOutput(output=code_output.output, is_final_answer=code_output.is_final_answer)

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
        return_full_result: bool | None = None,
        gt_answer: str | None = None,
    ) -> Any | RunResult:
        """
        Run the agent for the given task with caching capabilities.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in streaming mode.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.
            additional_args (`dict`, *optional*): Any other variables that you want to pass to the agent run.
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task.
            return_full_result (`bool`, *optional*): Whether to return the full [`RunResult`] object or just the final answer output.
            gt_answer (`str`, *optional*): Ground truth answer for validation. If provided, cache templates will only be created if the agent's output is correct.

        Example:
        ```py
        from smolagents import CachedCodeAgent
        agent = CachedCodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?", gt_answer="13.34")
        ```
        """
        # Store ground truth answer for validation
        self.gt_answer = gt_answer
        
        # Call parent run method
        result = super().run(
            task=task,
            stream=stream,
            reset=reset,
            images=images,
            additional_args=additional_args,
            max_steps=max_steps,
            return_full_result=return_full_result,
        )
        
        # Handle caching after the run is complete
        if not stream and hasattr(self, 'memory'):
            self._handle_post_run_caching(task, result)
        
        return result

    def _handle_post_run_caching(self, task: str, result: Any):
        """Handle caching logic after the run is complete."""
        try:
            final_answer = str(result) if result is not None else ""
            self._cache_plan_template(task, final_answer)
        except Exception as e:
            self.logger.log(f"Error in post-run caching: {e}", level=LogLevel.WARNING)

    def _generate_planning_step(self, task, is_first_step: bool, step: int) -> Generator[ChatMessageStreamDelta | PlanningStep]:
        """
        Override planning step with caching capabilities.
        For the first step, check cache for similar plan templates and adapt them.
        """
        if is_first_step:
            # Try to get a cached plan template
            print(f"$$$$$ Trying to get a cached plan template for task!")
            keyword, cached_template, keyword_tokens, latency_info = self.cache.extract_keyword_and_search_for_hit(task)
            
            # Track tokens used for keyword extraction
            self._track_small_model_tokens(keyword_tokens.input_tokens, keyword_tokens.output_tokens)
            
            if cached_template:
                print(f"$$$$$ Found a cached plan template for task")
                # Use cached plan - create planning step directly
                planning_step = self._create_cached_planning_step(task, cached_template)
                yield planning_step
                return
        
        print(f"$$$$$ No cached plan template found for task, falling back to standard planning")
        # Fall back to standard planning if no cache hit or not first step
        planning_step = None
        
        # Track tokens for large model planning
        start_time = time.time()
        if is_first_step:
            input_messages = [
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["initial_plan"],
                                variables={"task": task, "tools": self.tools, "managed_agents": self.managed_agents},
                            ),
                        }
                    ],
                )
            ]
            
            # Use large model for planning and track tokens
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                for event in self.model.generate_stream(input_messages):
                    yield event
                    if isinstance(event, ChatMessageStreamDelta):
                        plan_message_content += event.content
                plan_message = ChatMessage(role=MessageRole.ASSISTANT, content=plan_message_content)
            else:
                plan_message = self.model.generate(input_messages)
                
            # Track large model tokens
            if hasattr(plan_message, 'token_usage') and plan_message.token_usage:
                self._track_large_model_tokens(
                    plan_message.token_usage.input_tokens,
                    plan_message.token_usage.output_tokens
                )
            
            planning_step = PlanningStep(
                timing=Timing(start_time=start_time),
                model_input_messages=input_messages,
                model_output_message=plan_message,
                plan=plan_message.content,
                token_usage=plan_message.token_usage if hasattr(plan_message, 'token_usage') else None,
            )
            yield planning_step
        else:
            # For non-first steps, use the parent method
            for element in super()._generate_planning_step(task, is_first_step, step):
                yield element
                if isinstance(element, PlanningStep):
                    planning_step = element
                    # Track tokens if available
                    if hasattr(element, 'token_usage') and element.token_usage:
                        self._track_large_model_tokens(
                            element.token_usage.input_tokens,
                            element.token_usage.output_tokens
                        )

    def _create_cached_planning_step(self, task: str, cached_template) -> PlanningStep:
        """
        Create a planning step from a cached template.
        This is a simpler, more direct approach.
        """
        start_time = time.time()
        
        # Adapt the cached template to the current task
        adapted_plan, adaptation_tokens = self._adapt_cached_plan(cached_template, task)
        
        # Track the small model tokens used for adaptation
        self._track_small_model_tokens(adaptation_tokens.input_tokens, adaptation_tokens.output_tokens)
        
        # Create input messages for the adapted plan
        input_messages = [
            ChatMessage(
                role=MessageRole.USER,
                content=[{
                    "type": "text",
                    "text": adapted_plan,
                }]
            )
        ]
        
        # Generate the planning step with adapted content
        plan_message_content = adapted_plan
        # Use the actual tokens from adaptation instead of 0
        input_tokens, output_tokens = adaptation_tokens.input_tokens, adaptation_tokens.output_tokens
        
        plan = textwrap.dedent(
            f"""Here are the facts I know and the plan of action that I will follow to solve the task:\n```\n{plan_message_content}\n```"""
        )
        
        log_headline = "Cached plan (adapted)"
        self.logger.log(Rule(f"[bold]{log_headline}", style="green"), Text(plan), level=LogLevel.INFO)
        
        # Create and return the planning step directly
        planning_step = PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=plan_message_content),
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            timing=Timing(start_time=start_time, end_time=time.time()),
        )
        
        # Cache the adapted plan for future use
        self._cache_generated_plan(task, planning_step)
        
        return planning_step

    def _adapt_cached_plan(self, cached_template, current_task: str) -> tuple[str, TokenUsage]:
        """
        Adapt a cached plan template to the current task using a lightweight model.
        Similar to _get_initial_supervisor_response in minion framework.
        Returns both the adapted plan and token usage.
        """
        # Extract the plan template from the cached template
        
        # Create adaptation prompt for lightweight model
        adaptation_prompt = f"""
Please adapt the following reference plan template to work for the current research task.

Reference plan template: {cached_template}

Your task is to adapt the reference plan template to the current question, maintaining the same high-level structure and approach but customizing it for the specific details of the current question. Keep the plan concise and focused.

Current task: {current_task}

Return the adapted plan in the same GAIA format, including:
1. Facts survey (1.1 Facts given in the task, 1.2 Facts to look up, 1.3 Facts to derive)
2. Step-by-step plan

Make sure the adapted plan is specific to the current task but follows the same reasoning pattern as the original.

Adapted plan:
"""
        
        # Use lightweight model to adapt the plan (similar to minion framework)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using small model
                messages=[
                    {
                        "role": "user",
                        "content": adaptation_prompt
                    }
                ]
            )
            
            adapted_plan = response.choices[0].message.content.strip()
            token_usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
            return adapted_plan, token_usage
            
        except Exception as e:
            self.logger.log(f"Error adapting cached plan: {e}", level=LogLevel.WARNING)
            # Fall back to generating a basic plan
            fallback_plan = self._generate_fallback_plan(current_task)
            # Return minimal token usage for fallback
            return fallback_plan, TokenUsage(input_tokens=0, output_tokens=0)

    def _generate_fallback_plan(self, task: str) -> str:
        """
        Generate a basic fallback plan when adaptation fails.
        """
        return f"""
## 1. Facts survey
### 1.1. Facts given in the task
{task}

### 1.2. Facts to look up
Information needed to solve the task.

### 1.3. Facts to derive
Results and conclusions to be computed.

## 2. Plan
1. Analyze the task requirements
2. Gather necessary information
3. Process and compute results
4. Provide final answer
<end_plan>
"""

    def _cache_plan_template(self, task: str, final_answer: str = None):
        """
        Cache the plan template for future use, but only if the answer is correct.
        """
        try:
            # Check if we have a ground truth answer and if the final answer is correct
            if self.gt_answer and final_answer:
                if final_answer != self.gt_answer:
                    self.logger.log("Skipping cache induction: answer is incorrect", level=LogLevel.INFO)
                    return
            
            # Extract keyword for caching
            keyword, keyword_tokens = self.cache.extract_keyword(task)
            print(f"$$$$$ Extracted keyword for caching conversation: {keyword}")
            
            # Track tokens used for keyword extraction
            self._track_small_model_tokens(keyword_tokens.input_tokens, keyword_tokens.output_tokens)
            
            # Create conversation log structure from entire agent memory
            conversation_log = {
                "conversation": []
            }
            
            # Convert agent memory to conversation format
            for step in self.memory.steps:
                if isinstance(step, PlanningStep):
                    conversation_log["conversation"].append({
                        "user": "remote",
                        "output": json.dumps({
                            "message": step.plan,
                            "type": "planning"
                        })
                    })
                elif isinstance(step, ActionStep):
                    if step.model_output:
                        conversation_log["conversation"].append({
                            "user": "local", 
                            "output": step.model_output
                        })
                    if step.observations:
                        conversation_log["conversation"].append({
                            "user": "remote",
                            "output": json.dumps({
                                "message": step.observations,
                                "type": "observation"
                            })
                        })
                elif isinstance(step, FinalAnswerStep):
                    conversation_log["conversation"].append({
                        "user": "remote",
                        "output": json.dumps({
                            "answer": str(step.output),
                            "type": "final_answer"
                        })
                    })
            
            # Cache the template using induce_and_insert
            cachegen_tokens, plan_template = self.cache.induce_and_insert(task, keyword, conversation_log)
            
            # Track tokens used for plan template generation
            self._track_small_model_tokens(cachegen_tokens.input_tokens, cachegen_tokens.output_tokens)
            
        except Exception as e:
            self.logger.log(f"Error caching conversation history: {e}", level=LogLevel.WARNING)



class GAIAPlanCache:
    """
    Cache for GAIA plan templates with keyword-based lookup and semantic similarity matching.
    """
    
    def __init__(self, cache_size_cap):
        self.cache_size_cap = cache_size_cap
        self.cache_size = 0
        self.cache = {}
        self.cachegen_overhead = {"input": 0, "output": 0}
        self.access_order = OrderedDict() 
        self.static_sim_score_threshold = 0.4
        
        # Initialize OpenAI client for keyword extraction
        self.client = OpenAI()
        
        # Initialize similarity model
        try:
            from sentence_transformers import SentenceTransformer, util
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.util = util
        except ImportError:
            self.similarity_model = None
            self.util = None

    def filter_keyword(self, raw_keyword):
        # Convert to lowercase
        raw_keyword = raw_keyword.lower()
        
        # Remove "(...)" patterns
        raw_keyword = re.sub(r"\s*\(.*?\)\s*", "", raw_keyword)
        
        # Remove punctuation except spaces
        raw_keyword = re.sub(r"[^\w\s]", "", raw_keyword)
        
        # Normalize spaces
        raw_keyword = re.sub(r"\s+", " ", raw_keyword).strip()
        
        return raw_keyword

    def semantic_similarity(self, prompt1, prompt2):
        """Compute semantic similarity between two prompts."""
        if not self.similarity_model:
            # Fallback to simple word overlap
            words1 = set(prompt1.lower().split())
            words2 = set(prompt2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0
        
        try:
            embedding1 = self.similarity_model.encode(prompt1, convert_to_tensor=True)
            embedding2 = self.similarity_model.encode(prompt2, convert_to_tensor=True)
            similarity = self.util.cos_sim(embedding1, embedding2)
            return similarity.item()
        except Exception:
            return 0.0

    def extract_task(self, query):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"Can you help me summarize what is the 'task' / 'keyword' behind this query (Please answer only with the task / keyword, must be independent from problem-specific details)?\n{query}",
                }
            ]
        )
        token_usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )
        raw_task = response.choices[0].message.content
        filtered_task = self.filter_keyword(raw_task)
        return filtered_task, token_usage

    def extract_keyword(self, query):
        """Extract keyword from GAIA research task."""
        keyword_extraction_prompt = f"""
        Can you tell me the most important 'keyword' for the following research question (Please answer only with the keyword, must be independent from specific details)?
        {query}
        Example: 'What is the population of Paris in 2023?' -> 'population research'.
        Example: 'Calculate the GDP growth rate of France from 2020 to 2023' -> 'economic calculation'.
        Example: 'Find the average temperature in Tokyo during summer' -> 'climate data analysis'.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": keyword_extraction_prompt,
                }
            ]
        )
        token_usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )
        raw_keyword = response.choices[0].message.content
        filtered_keyword = self.filter_keyword(raw_keyword)
        return filtered_keyword, token_usage

    def search_for_hit(self, query, threshold=0.6):
        """Search for cached plan templates similar to the query."""
        if self.cache == {}:
            return None, 0.0

        # for exact match mode, return the first match entry
        if threshold == "exact match":
            for key, value in self.cache.items():
                if key == query:
                    if key in self.access_order:
                        self.access_order.move_to_end(key)
                    return value, 1.0
            return None, 0.0

        max_sim, max_item = 0.0, (None, None)
        for key, value in self.cache.items():
            similarity = self.semantic_similarity(query, key)
            if similarity > max_sim:
                max_sim = similarity
                max_item = (key, value)
        if max_sim > threshold:
            key = max_item[0]
            if key in self.access_order:
                self.access_order.move_to_end(key)
            print(f"retrieved key: {key}")
            print(f"max sim: {max_sim}")
            return max_item[1], max_sim
        else:
            return None, 0.0

    def extract_keyword_and_search_for_hit(self, query, threshold=None):
        """Extract keyword and search for cached plan templates."""
        latency_info = {}
        total_token_usage = TokenUsage(input_tokens=0, output_tokens=0)
        
        # extract keyword
        keyword_p1 = time.time()
        keyword, keyword_tokens = self.extract_keyword(query)
        total_token_usage.input_tokens += keyword_tokens.input_tokens
        total_token_usage.output_tokens += keyword_tokens.output_tokens
        print(f"$$$$$ Extracted keyword: {keyword}")
        keyword_p2 = time.time()
        
        # search for hit
        lookup_p1 = time.time()
        if threshold is None:
            value, sim_score = self.search_for_hit(keyword, threshold="exact match")
        else:
            value, sim_score = self.search_for_hit(keyword, threshold=threshold)
        lookup_p2 = time.time()
        latency_info["extract_keyword"] = keyword_p2 - keyword_p1
        latency_info["cache_lookup"] = lookup_p2 - lookup_p1
        return keyword, copy.deepcopy(value), total_token_usage, latency_info



    def insert(self, key, value):
        """Insert a plan template into the cache."""
        if self.cache_size_cap == 0:
            return 
        
        # insert
        self.cache[key] = value
        self.access_order[key] = None
        self.cache_size += 1
        
        # evict
        if self.cache_size > self.cache_size_cap:
            self.evict(strategy="LRU")

    def induce_and_insert(self, query, keyword, conversation_log, query_as_key=False):
        """Extract and cache a plan template from successful GAIA execution."""
        # Extract plan template from the conversation log
        plan_template = self._extract_plan_template(query, keyword, conversation_log)
        
        
        # Validate the plan template
        if not self._validate_plan_template(plan_template):
            print("Warning: Plan template doesn't follow expected patterns. Applying fixes.")
            plan_template = self._fix_plan_template(plan_template)
        plan_template_str = plan_template.get("plan_template", "")

        # Create cache induction prompt for GAIA plans
        cache_inducing_prompt = """
            You will see a JSON trace that shows the complete planning workflow of how a GAIA research agent solves a complex task. Clean up the plan template so that we can reuse this trace as a reference template (independent from specific details like company names, dates, or specific values) when we meet similar research tasks later. 
            
            Requirements: 
            (1) the plan should follow the GAIA structure: Facts survey (1.1 Facts given, 1.2 Facts to look up, 1.3 Facts to derive) and Plan (step-by-step approach),
            (2) the task and the plan should not contain problem-specific details or numbers, and 
            (3) return the result in JSON format that can be parsed by Python's json.loads().

            IMPORTANT: The plan must maintain the GAIA planning structure to ensure proper functioning. Always include both the facts survey and step-by-step plan sections.
            
            JSON trace:
        """
        
        # Add examples of well-structured GAIA plans
        examples = """
        Here is one example of a properly structured GAIA plan:
        
        {
            "keyword": "population research",
            "task": "Research and calculate population statistics for a given city.",
            "plan_template": "## 1. Facts survey\n### 1.1. Facts given in the task\n[Task description]\n\n### 1.2. Facts to look up\nPopulation data, demographic information, statistical sources\n\n### 1.3. Facts to derive\nPopulation calculations, growth rates, demographic analysis\n\n## 2. Plan\n1) Gather population data from reliable sources\n2) Analyze demographic trends\n3) Calculate relevant statistics\n4) Verify results and provide final answer"
        }
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": cache_inducing_prompt + examples + "\n\nYour task:\n" + plan_template_str,
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            cachegen_token_usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
            self.cachegen_overhead["input"] += cachegen_token_usage.input_tokens
            self.cachegen_overhead["output"] += cachegen_token_usage.output_tokens
            
            try:
                # First try direct JSON parsing
                cached_template = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                # Fall back to extraction if needed
                cached_template = self._extract_json(response.choices[0].message.content)
            
            # Validate the structure of the induced template
            cached_template = self._validate_and_fix_template(cached_template)
            print(f"$$$$$ Cached plan template: {cached_template}")
            
            # Insert validated template into cache
            if query_as_key:
                self.insert(query, cached_template)
            else:
                self.insert(keyword, cached_template)
            
            return cachegen_token_usage, plan_template
            
        except Exception as e:
            print(f"Error during cache induction: {str(e)}")
            # Create a simple valid template as fallback
            fallback_template = {
                "keyword": keyword,
                "task": query,
                "plan_template": "## 1. Facts survey\n### 1.1. Facts given in the task\n[Task description]\n\n### 1.2. Facts to look up\nRelevant data and information sources\n\n### 1.3. Facts to derive\nResults and conclusions to be computed\n\n## 2. Plan\n1. Analyze the task requirements\n2. Gather necessary information\n3. Process and compute results\n4. Provide final answer"
            }
            if query_as_key:
                self.insert(query, fallback_template)
            else:
                self.insert(keyword, fallback_template)
            return {"input": 0, "output": 0}, plan_template

    def _extract_plan_template(self, query, keyword, conversation_log):
        """Extract plan template from GAIA conversation log."""
        # For GAIA, we extract the planning step content from conversation format
        # TODO: build a GAIA plan from the entire conversation, not just the initial plan
        print(f"$$$$$ Extracting plan template from conversation log")
        plan_content = ""
        
        # Look for planning steps in the conversation log
        for conversation_entry in conversation_log.get("conversation", []):
            if conversation_entry.get("user") == "remote":
                try:
                    output_data = json.loads(conversation_entry.get("output", "{}"))
                    if output_data.get("type") == "planning":
                        plan_content = output_data.get("message", "")
                        break
                except json.JSONDecodeError:
                    # If not JSON, check if it's a plain planning message
                    if "planning" in conversation_entry.get("output", "").lower():
                        plan_content = conversation_entry.get("output", "")
                        break
        
        # If no planning step found, create a basic template
        if not plan_content:
            plan_content = f"## 1. Facts survey\n### 1.1. Facts given in the task\n{query}\n\n### 1.2. Facts to look up\nRelevant data and information sources\n\n### 1.3. Facts to derive\nResults and conclusions to be computed\n\n## 2. Plan\n1. Analyze the task requirements\n2. Gather necessary information\n3. Process and compute results\n4. Provide final answer"
        
        print(f"$$$$$ Extracted plan template: {plan_content}")
        return {
            "keyword": keyword,
            "task": query,
            "plan_template": plan_content
        }

    def _validate_plan_template(self, plan_template):
        """Validate that a plan template follows GAIA structure."""
        plan_content = plan_template.get("plan_template", "")
        
        # Check for required sections
        required_sections = ["## 1. Facts survey", "## 2. Plan"]
        for section in required_sections:
            if section not in plan_content:
                return False
        
        # Check for facts survey subsections
        facts_subsections = ["### 1.1.", "### 1.2.", "### 1.3."]
        for subsection in facts_subsections:
            if subsection not in plan_content:
                return False
        
        return True

    def _fix_plan_template(self, plan_template):
        """Fix a plan template that doesn't follow the expected structure."""
        plan_content = plan_template.get("plan_template", "")
        
        # Ensure we have the basic structure
        if "## 1. Facts survey" not in plan_content:
            plan_content = "## 1. Facts survey\n" + plan_content
        
        if "## 2. Plan" not in plan_content:
            plan_content += "\n\n## 2. Plan\n1. Analyze the task requirements\n2. Gather necessary information\n3. Process and compute results\n4. Provide final answer"
        
        # Ensure facts survey subsections exist
        if "### 1.1." not in plan_content:
            plan_content = plan_content.replace("## 1. Facts survey", "## 1. Facts survey\n### 1.1. Facts given in the task\n[Task description]")
        
        if "### 1.2." not in plan_content:
            plan_content = plan_content.replace("### 1.1.", "### 1.1. Facts given in the task\n[Task description]\n\n### 1.2. Facts to look up\nRelevant data and information sources")
        
        if "### 1.3." not in plan_content:
            plan_content = plan_content.replace("### 1.2.", "### 1.2. Facts to look up\nRelevant data and information sources\n\n### 1.3. Facts to derive\nResults and conclusions to be computed")
        
        return {
            "keyword": plan_template.get("keyword", "unknown"),
            "task": plan_template.get("task", "Analyze the information and provide an answer."),
            "plan_template": plan_content
        }

    def _validate_and_fix_template(self, template):
        """Validate and fix the structure of a generated template."""
        # Ensure required fields exist
        if "keyword" not in template:
            template["keyword"] = "unknown"
        if "task" not in template:
            template["task"] = "Analyze the information and provide an answer."
        if "plan_template" not in template:
            template["plan_template"] = "## 1. Facts survey\n### 1.1. Facts given in the task\n[Task description]\n\n### 1.2. Facts to look up\nRelevant data and information sources\n\n### 1.3. Facts to derive\nResults and conclusions to be computed\n\n## 2. Plan\n1. Analyze the task requirements\n2. Gather necessary information\n3. Process and compute results\n4. Provide final answer"
        
        # Validate plan template structure
        if not self._validate_plan_template(template):
            template = self._fix_plan_template(template)
        
        return template

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from text that may be wrapped in markdown code blocks."""
        block_matches = list(re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL))
        bracket_matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))

        if block_matches:
            json_str = block_matches[-1].group(1).strip()
        elif bracket_matches:
            json_str = bracket_matches[-1].group(0)
        else:
            json_str = text

        # Escape newlines within quoted JSON strings
        json_str = self._escape_newlines_in_strings(json_str)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {json_str}")
            raise

    def _escape_newlines_in_strings(self, json_str: str) -> str:
        """Replace literal newlines in JSON string values with escaped newlines."""
        return re.sub(
            r'(".*?")',
            lambda m: m.group(1).replace("\n", "\\n"),
            json_str,
            flags=re.DOTALL,
        )

    def evict(self, strategy):
        """Evict items from cache using the specified strategy."""
        if strategy == "LRU":
            # Remove least recently used item (first item in OrderedDict)
            lru_key, _ = self.access_order.popitem(last=False)
            del self.cache[lru_key]
            self.cache_size -= 1
        else:
            raise NotImplementedError(f"Strategy {strategy} not implemented")