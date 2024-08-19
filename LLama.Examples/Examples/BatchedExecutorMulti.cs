using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Nodes;
using LLama.Grammars;
using LLama.Batched;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using Spectre.Console;
using System.Diagnostics;
using DocumentFormat.OpenXml.Spreadsheet;
using Microsoft.Identity.Client;
using static Microsoft.KernelMemory.Constants.CustomContext;

namespace LLama.Examples.Examples;


public class OutlineTopLevel
{
    [JsonPropertyName("topic_sentence")]
    public string TopicSentence { get; set; } = string.Empty;

    [JsonPropertyName("main_points")]
    public string[] MainPoints { get; set; } = [];

    public bool HasAllData()
    {
        var hasTopicSentence = !string.IsNullOrEmpty(TopicSentence);
        if (!hasTopicSentence)
        {
            return false;
        }
        if ((MainPoints == null) || (MainPoints.Length < 3))
        {
            return false;
        }
        for(var index = 0; index < MainPoints.Length; index++)
        {
            if (string.IsNullOrEmpty(MainPoints[index]))
            {
                return false;
            }
        }
        return true;
    }
}

    public class OutlineMainPoint
{
    [JsonPropertyName("main_point_summary")]
    public string MainPointSummary { get; set; } = string.Empty;

    [JsonPropertyName("supporting_points")]
    public string[] SupportingPoints { get; set; } = [];


    public bool HasAllData()
    {
        var hasMainPointSummary = !string.IsNullOrEmpty(MainPointSummary);
        if (!hasMainPointSummary)
        {
            return false;
        }
        if ((SupportingPoints == null) || (SupportingPoints.Length < 3))
        {
            return false;
        }
        for (var index = 0; index < SupportingPoints.Length; index++)
        {
            if (string.IsNullOrEmpty(SupportingPoints[index]))
            {
                return false;
            }
        }
        return true;
    }
}

public class GrammarParser
{
    private static GrammarParser instance;
    public static GrammarParser Instance
    {
        get
        {
            if (instance == null)
            {
                instance = new GrammarParser();
            }
            return instance;
        }
    }

    private const string GRAMMAR_ROOT_NODE = "root";

    private readonly Dictionary<string, SafeLLamaGrammarHandle> mapTextToGrammarHandle = new();


    /// <summary>
    /// Parse grammar. Json->Json Schema->Json Grammar. Pass the grammar to this function which will generate a list of GrammarRules which 
    /// can be passed in the chat InferenceParams.
    /// See also Project Llama.cpp: test-grammar-integration.cpp::build_grammar()
    /// online tool: https://adrienbrault.github.io/json-schema-to-gbnf/
    /// llama.cpp: https://github.com/ggerganov/llama.cpp
    /// </summary>
    /// <param name="grammarText"></param>
    /// <returns>clone of the grammar's safe handle</returns>
    public SafeLLamaGrammarHandle ParseGrammar(string grammarText)
    {
        if (mapTextToGrammarHandle.ContainsKey(grammarText))
        {
            //Debug.Log($"ParseGrammar(): Found grammar. Returning clone of handle...");
            return mapTextToGrammarHandle[grammarText].Clone();
        }

        var parsedGrammar = Grammar.Parse(grammarText, GRAMMAR_ROOT_NODE);

        Debug.Assert(parsedGrammar.Rules.Count > 0, "ParseGrammar(): No Grammar Rules found!");

        // Search for the root rule so that we can .
        for (var index = 0; index < parsedGrammar.Rules.Count; index++)
        {
            if (parsedGrammar.Rules[index].Name.CompareTo(GRAMMAR_ROOT_NODE) == 0)
            {
                // Create the handle with an index to the root rule of the grammar.
                var handle = SafeLLamaGrammarHandle.Create(parsedGrammar.Rules, (ulong)index);
                mapTextToGrammarHandle[grammarText] = handle;
                return handle.Clone();
            }
        }

        //Debug.LogError($"ParseGrammar(): Grammar doesn't have \"{GRAMMAR_ROOT_NODE}\" node!");
        return null;
    }

    public void ReleaseGrammarHandle(SafeLlamaModelHandle handle)
    {
        handle.Close();
    }
}

public class RandomNumberGenerator
{
    private Random _random;

    // Constructor to seed the random number generator
    public RandomNumberGenerator()
    {
        // Get the current time in ticks
        long ticks = DateTime.Now.Ticks;

        // Compute a hash of the ticks to use as a seed
        int seed = ticks.GetHashCode();

        // Initialize the Random object with the seed
        _random = new Random(seed);
    }

    // Function to generate a random integer within a specified range [min, max)
    public int RandomRange(int min, int max)
    {
        if (min >= max)
        {
            throw new ArgumentException("min must be less than max");
        }

        return _random.Next(min, max);
    }
}


/// <summary>
/// This demonstrates using a batch to generate two sequences and then using one
/// sequence as the negative guidance ("classifier free guidance") for the other.
/// </summary>
public class BatchedExecutorMultiGuidance
{
    //private const int TokenCount = 200;

    private const string SYSTEM_PROMPT = "Perform the task to the best of your ability.";
    private const string INITIAL_PROMPT_FORMAT = "<s>[INST] {0} {1} {2} [/INST] ";
    private const string FOLLOW_UP_FORMAT = "{0}</s>[INST] {1} [/INST]";

    private const string OUTLINE_FORMAT = "**OUTLINE FORMAT**\r\n\r\nUse Roman numerals, capital letters, and Arabic numerals for different levels of information:\r\n\r\nI. Introduction\r\n   A. Background information on the topic\r\n   B. Thesis statement or central argument\r\n   C. Overview of the main points to be discussed\r\n\r\nII. Body Paragraph 1\r\n   A. Topic sentence introducing the main point\r\n   B. Supporting evidence (facts, statistics, examples)\r\n   C. Explanation or analysis of the evidence\r\n   D. Transition to the next point\r\n\r\nIII. Body Paragraph 2\r\n   A. Topic sentence introducing the next main point\r\n   B. Supporting evidence (facts, statistics, examples)\r\n   C. Explanation or analysis of the evidence\r\n   D. Transition to the next point\r\n\r\nIV. Body Paragraph 3 (Optional)\r\n   A. Topic sentence introducing the final main point\r\n   B. Supporting evidence (facts, statistics, examples)\r\n   C. Explanation or analysis of the evidence\r\n   D. Transition to the conclusion\r\n\r\nV. Conclusion\r\n   A. Restatement of the thesis statement\r\n   B. Summary of the main points\r\n   C. Final thoughts or concluding remarks.";

    private const string OUTLINE_TOP_LEVEL_JSON = "{\r\n    \"topic_sentence\": \"Topic Sentence\",\r\n    \"main_points\": [\r\n        \"Main Point 1\",\r\n        \"Main Point 2\",\r\n        \"Main Point 3\"\r\n    ]\r\n}";
    private const string OUTLINE_TOP_LEVEL_GRAMMAR = "root ::= Outline\r\nOutline ::= \"{\"   ws   \"\\\"topic_sentence\\\":\"   ws   string   \",\"   ws   \"\\\"main_points\\\":\"   ws   stringlist   \"}\"\r\nOutlinelist ::= \"[]\" | \"[\"   ws   Outline   (\",\"   ws   Outline)*   \"]\"\r\nstring ::= \"\\\"\"   ([^\"]*)   \"\\\"\"\r\nboolean ::= \"true\" | \"false\"\r\nws ::= [ \\t\\n]*\r\nnumber ::= [0-9]+   \".\"?   [0-9]*\r\nstringlist ::= \"[\"   ws   \"]\" | \"[\"   ws   string   (\",\"   ws   string)*   ws   \"]\"\r\nnumberlist ::= \"[\"   ws   \"]\" | \"[\"   ws   string   (\",\"   ws   number)*   ws   \"]\"\r\n";

    private const string OUTLINE_MAIN_POINT_JSON = "{\r\n    \"main_point_summary\": \"Main Point Summary\",\r\n    \"supporting_points\": [\r\n        \"Supporting Point 1\",\r\n        \"Supporting Point 2\",\r\n        \"Supporting Point 3\"\r\n    ]\r\n}";
    private const string OUTLINE_MAIN_POINT_GRAMMAR = "root ::= \"{\" ws01 root-main-point-summary \",\" ws01 root-supporting-points \"}\" ws01\r\nroot-main-point-summary ::= \"\\\"main_point_summary\\\"\" \":\" ws01 string\r\nroot-supporting-points ::= \"\\\"supporting_points\\\"\" \":\" ws01 \"[\" ws01 (root-supporting-points-items (ws01 \",\" ws01 root-supporting-points-items)*)? ws01 \"]\"\r\nroot-supporting-points-items ::= string\r\n\r\n\r\nvalue  ::= (object | array | string | number | boolean | null) ws\r\n\r\nobject ::=\r\n  \"{\" ws (\r\n    string \":\" ws value\r\n    (\",\" ws string \":\" ws value)*\r\n  )? \"}\"\r\n\r\narray  ::=\r\n  \"[\" ws01 (\r\n            value\r\n    (\",\" ws01 value)*\r\n  )? \"]\"\r\n\r\nstring ::=\r\n  \"\\\"\" (string-char)* \"\\\"\"\r\n\r\nstring-char ::= [^\"\\\\] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes\r\n\r\nnumber ::= integer (\".\" [0-9]+)? ([eE] [-+]? [0-9]+)?\r\ninteger ::= \"-\"? ([0-9] | [1-9] [0-9]*)\r\nboolean ::= \"true\" | \"false\"\r\nnull ::= \"null\"\r\n\r\n# Optional space: by convention, applied in this grammar after literal chars when allowed\r\nws ::= ([ \\t\\n] ws)?\r\nws01 ::= ([ \\t\\n])?";

    private const int MAX_CONVERSATION_TOKEN_COUNT = 30;
    private const int MAX_BATCH_TOKEN_COUNT = 100; //10000;

    private const int NUM_BATCHED_CONVERSATIONS = 1; //64;

    private const int MAX_NUM_RETRIES = 10;

    private const float DEFAULT_TEMPERATURE = 0.3f;


    // Default Inference Parameters
    public const float DEFAULT_LLM_TEMPERATURE = 0.45f; //0.8f;  // [0..1]
    public const int DEFAULT_LLM_MAX_TOKENS = -1;       // -1 == no limit
    public const int DEFAULT_TOP_K = 40;                // Choose from top K tokens
    public const float DEFAULT_REPEAT_PENALTY = 1.0f;
    public const float DEFAULT_MIN_P = 0.02f; //0.05f;
    public const float DEFAULT_TOP_P = 1.0f; //0.95f;

    private static RandomNumberGenerator random = new();

    private static bool TotalBatchTokensReached => (BatchTokenCount >= MAX_BATCH_TOKEN_COUNT);
    private static int BatchTokenCount { get; set; }

    private const DecodeResult DEFAULT_INFERENCE_RESULT = DecodeResult.Ok;
    private static DecodeResult InferenceDecodeResult { get; set; } = DEFAULT_INFERENCE_RESULT;
    private static bool InferenceDecodeErrorFound => InferenceDecodeResult != DecodeResult.Ok;

    private static readonly Stopwatch timer = Stopwatch.StartNew();
    private static DecodeResult decodeResult = DecodeResult.Ok;
    private static readonly List<string> errors = new();
    private static readonly Dictionary<int, InferenceResultData> mapConversationIdToInferenceData = new();
    private static ProgressTask reporter;
    private static BatchedExecutor executor;
    private static LLamaWeights model;
    private static int numIterations = 0;

    private static StatelessExecutor statelessExecutor;


    public enum InferenceStatus
    {
        GatherNotes,
        NotesComplete,
        WriteOutlineTopLevel,
        WriteOutlineMainPoint1,
        WriteOutlineMainPoint2,
        WriteOutlineMainPoint3,
        WriteFirstDraft,
        WriteFinalVersion,

        Error,
        MaxConversationTokensReached
    }

    public class InferenceResultData(int id, int maxTokens, string userMsg, Conversation conversation, BaseSamplingPipeline samplingPipeline, StreamingTokenDecoder streamingTokenDecoder)
    {
        public int Id { get; set; } = id;
        public InferenceStatus Status { get; set; } = InferenceStatus.GatherNotes;
        public int NumTokens { get; set; } = 0;
        public int MaxTokens { get; set; } = maxTokens;
        public int NumErrors { get; set; } = 0;
        public int TotalTokens { get; set; } = 0;
        public DecodeResult KvCacheResult { get; set; } = DecodeResult.Ok;
        public string UserMsg { get; set; } = userMsg;
        public List<string> Response { get; set; } = new();
        public string OutlineResponse { get; set; } = string.Empty;
        public string FirstDraft { get; set; } = string.Empty;
        public string EditedResponse { get; set; } = string.Empty;
        public OutlineTopLevel OutlineTopLevel { get; set; } = new();
        public Dictionary<int, OutlineMainPoint> OutlineMainPoint { get; set; } = new();

        public Conversation Conversation { get; set; } = conversation;
        public BaseSamplingPipeline SamplingPipeline { get; set; } = samplingPipeline;
        public StreamingTokenDecoder StreamingTokenDecoder { get; set; } = streamingTokenDecoder;

        public void DebugWriteOutline()
        {
            AnsiConsole.WriteLine($"Q: {UserMsg}\n");

            AnsiConsole.WriteLine($"Topic Sentence: {OutlineTopLevel.TopicSentence}\n");

            AnsiConsole.WriteLine($"I. Main Point 1: {OutlineTopLevel.MainPoints[0]}\n");
            AnsiConsole.WriteLine($"      Summary: {OutlineMainPoint[0].MainPointSummary}\n");
            AnsiConsole.WriteLine($"   A. Supporting Point 1.1: {OutlineMainPoint[0].SupportingPoints[0]}");
            AnsiConsole.WriteLine($"   B. Supporting Point 1.2: {OutlineMainPoint[0].SupportingPoints[1]}");
            AnsiConsole.WriteLine($"   C. Supporting Point 1.3: {OutlineMainPoint[0].SupportingPoints[2]}\n");

            AnsiConsole.WriteLine($"II. Main Point 2: {OutlineTopLevel.MainPoints[1]}\n");
            AnsiConsole.WriteLine($"      Summary: {OutlineMainPoint[1].MainPointSummary}\n");
            AnsiConsole.WriteLine($"   A. Supporting Point 2.1: {OutlineMainPoint[1].SupportingPoints[0]}");
            AnsiConsole.WriteLine($"   B. Supporting Point 2.2: {OutlineMainPoint[1].SupportingPoints[1]}");
            AnsiConsole.WriteLine($"   C. Supporting Point 2.3: {OutlineMainPoint[1].SupportingPoints[2]}\n");

            AnsiConsole.WriteLine($"III. Main Point 3: {OutlineTopLevel.MainPoints[2]}\n");
            AnsiConsole.WriteLine($"      Summary: {OutlineMainPoint[2].MainPointSummary}\n");
            AnsiConsole.WriteLine($"   A. Supporting Point 3.1: {OutlineMainPoint[2].SupportingPoints[0]}");
            AnsiConsole.WriteLine($"   B. Supporting Point 3.2: {OutlineMainPoint[2].SupportingPoints[1]}");
            AnsiConsole.WriteLine($"   C. Supporting Point 3.3: {OutlineMainPoint[2].SupportingPoints[2]}\n");
        }
    }

    public static async Task Run()
    {
        await Init();

        // Run inference loop
        numIterations++;
        await AnsiConsole
           .Progress()
           .StartAsync(async progress =>
            {
                reporter = progress.AddTask($"Running Inference {numIterations}", maxValue: MAX_CONVERSATION_TOKEN_COUNT);

                await RunBatchedInference();
           }
        );

        ReportResults();

        while (await Continue())
        {
            // Run inference loop
            numIterations++;
            await AnsiConsole
               .Progress()
               .StartAsync(async progress =>
               {
                   reporter = progress.AddTask($"Running Inference {numIterations}", maxValue: MAX_CONVERSATION_TOKEN_COUNT);

                   await RunBatchedInference();
               }
            );

            ReportResults();
        }
        AnsiConsole.WriteLine("DONE.");
    }

    private static async Task Init()
    {
        AnsiConsole.WriteLine("BEGIN...");

        // Load model weights
        var parameters = new ModelParams(UserSettings.GetModelPath());
        parameters.GpuLayerCount = 35;
        //parameters.GpuLayerCount = 8;
        //parameters.NoKqvOffload = true;
        model = await LLamaWeights.LoadFromFileAsync(parameters);

        statelessExecutor = new StatelessExecutor(model, parameters);

        //var positivePrompt = AnsiConsole.Ask("Positive Prompt (or ENTER for default):", "My favourite colour is").Trim();
        //var negativePrompt = AnsiConsole.Ask("Negative Prompt (or ENTER for default):", "I hate the colour red. My favourite colour is").Trim();
        //var weight = AnsiConsole.Ask("Guidance Weight (or ENTER for default):", 2.0f);

        // Create an executor that can evaluate a batch of conversations together
        executor = new BatchedExecutor(model, parameters);

        executor.Context.SaveState("SaveState");

        // Print some info
        var name = model.Metadata.GetValueOrDefault("general.name", "unknown model name");
        Console.WriteLine($"Created executor with model: {name}");

        // Load the two prompts into two conversations
        //var guided = new List<Conversation>();
        //var guidedSampler = new List<DefaultSamplingPipeline>();
        //var guidedDecoder = new List<StreamingTokenDecoder>();
        //var guidedInProgress = new HashSet<int>();
        //var guidedNumTokens = new List<int>();

        //var activeConversations = new Queue<InferenceResultData>();
        mapConversationIdToInferenceData.Clear();

        // Init Random number generator
        new Random(questions.Length - 1);

        for (var index = 0; index < NUM_BATCHED_CONVERSATIONS; index++)
        {
            //var promptIndex = index % questions.Length;
            var promptIndex = random.RandomRange(0, questions.Length - 1);
            var currPrompt = questions[promptIndex];

            var currGuided = executor.Create();
            currGuided.Prompt(executor.Context.Tokenize(currPrompt));

            var samplingPipeline = new DefaultSamplingPipeline()
            {
                Temperature = DEFAULT_TEMPERATURE,
                PenalizeNewline = false,
                RepeatPenalty = DEFAULT_REPEAT_PENALTY,
                MinP = DEFAULT_MIN_P,
                TopP = DEFAULT_TOP_P,
                TopK = DEFAULT_TOP_K
            };
            var streamingTokenDecoder = new StreamingTokenDecoder(executor.Context);

            // Initialize the conversations that are active and the number of tokens they've received.
            mapConversationIdToInferenceData.Add(index, new(index, MAX_CONVERSATION_TOKEN_COUNT, currPrompt, currGuided, samplingPipeline, streamingTokenDecoder));
        }

        BatchTokenCount = 0; // guided and unguided.
        timer.Restart();
        decodeResult = DecodeResult.Ok;
        errors.Clear();
    }

    private static async Task<bool>Continue()
    {
        AnsiConsole.WriteLine("CONTINUE...");

        BatchTokenCount = 0;
        InferenceDecodeResult = DecodeResult.Ok;

        executor.Context.NativeHandle.KvCacheClear();
        var result = await executor.Infer();
        var numInProgress = 0;
        foreach (var kvp in mapConversationIdToInferenceData)
        {
            var currStatus = kvp.Value;
            if (currStatus.Status == InferenceStatus.WriteFinalVersion)
            {
                continue;
            }
            if (currStatus.Status == InferenceStatus.NotesComplete)
            {
                await RewriteResponse(currStatus);
                continue;
            }
            if (currStatus.Status == InferenceStatus.Error)
            {
                currStatus.NumErrors++;
                if (currStatus.NumErrors < 3)
                {
                    currStatus.Status = InferenceStatus.GatherNotes;
                }
            }
            if (currStatus.Status == InferenceStatus.Error)
            {
                await RewriteResponse(currStatus);
                continue;
            }
            currStatus.TotalTokens += currStatus.NumTokens;
            if (currStatus.TotalTokens >= MAX_CONVERSATION_TOKEN_COUNT * 10)
            {
                currStatus.Status = InferenceStatus.MaxConversationTokensReached;
                await RewriteResponse(currStatus);
                continue;
            }

            var currConversation = currStatus.Conversation;

            var concatenatedResponse = string.Concat(currStatus.Response);
            var prompt = $"<s>[INST] {SYSTEM_PROMPT} Carefully read the following Question and Answer. In your response, write what comes next.\nQuestion: \"{currStatus.UserMsg}\"\nAnswer: \"{concatenatedResponse}\" </s>";
            currStatus.Conversation.Prompt(executor.Context.Tokenize(prompt));

            currStatus.Status = InferenceStatus.GatherNotes;
            currStatus.NumTokens = 0;
            numInProgress++;
        }

        AnsiConsole.WriteLine($"Continue: numInProgress={numInProgress}");
        return (numInProgress > 0);
    }

    private static async Task RewriteResponse(InferenceResultData currStatus)
    {
        var concatenatedResponse = string.Concat(currStatus.Response);
        bool hasTopLevel = await WriteOutlineTopLevel(currStatus, concatenatedResponse);
        bool hasMainPoint1 = await WriteOutlineMainPoint(currStatus, concatenatedResponse, 0);
        bool hasMainPoint2 = await WriteOutlineMainPoint(currStatus, concatenatedResponse, 1);
        bool hasMainPoint3 = await WriteOutlineMainPoint(currStatus, concatenatedResponse, 2);

        if (hasTopLevel && hasMainPoint1 && hasMainPoint2 && hasMainPoint3)
        {
            currStatus.DebugWriteOutline();
        }

        //// Rewrite response using a stateless executor and put the result in currStatus.EditedResponse.
        ////var prompt = $"{SYSTEM_PROMPT} Rewrite the following paragraph and fix its grammar and prose so that it reads better. Write at a college-graduate level.\n*** Paragraph ***:{concatenatedResponse}";
        //var prompt = $"{SYSTEM_PROMPT}\n\n**CONTEXT**\n{concatenatedResponse}\n\n**USER**\n{currStatus.UserMsg}. Carefully and closely read the information provided in **CONTEXT* to write your response. Write at a college-graduate level.\n\n**ASSISTANT**\n";
        //var response = await statelessExecutor.InferAsync(prompt, statelessInferenceParams).ToListAsync();
        //currStatus.EditedResponse = string.Concat(response);
        //currStatus.Status = InferenceStatus.WriteFinalVersion;
    }



    private static async Task<bool> WriteOutlineTopLevel(InferenceResultData currStatus, string concatenatedResponse)
    {
        var prompt = $"{SYSTEM_PROMPT} Write your reponse using the following **JSON Format**\n{OUTLINE_TOP_LEVEL_JSON}\n\n**CONTEXT**\n{concatenatedResponse}\n\n**USER**\n{currStatus.UserMsg}. Carefully and closely read the information provided in **CONTEXT** and enumerate 3 MAIN POINTS. Then, Write a TOPIC SENTENCE by summarizing the 3 MAIN POINTS.\n\n**ASSISTANT**\n";

        var statelessInferenceParams = GetStatelessInferenceParams(OUTLINE_TOP_LEVEL_GRAMMAR);

        var numRetries = 0;
        var hasAllData = false;
        while ((numRetries < MAX_NUM_RETRIES) && !hasAllData)
        {
            AnsiConsole.WriteLine($"WriteOutlineTopLevel():  Try {numRetries + 1} of {MAX_NUM_RETRIES}");
            var response = await statelessExecutor.InferAsync(prompt, statelessInferenceParams).ToListAsync();
            currStatus.OutlineResponse = string.Concat(response);

            try
            {
                var outlineTopLevel = JsonSerializer.Deserialize<OutlineTopLevel>(currStatus.OutlineResponse);
                if ((outlineTopLevel != null) && (hasAllData = outlineTopLevel.HasAllData()))
                {
                    currStatus.OutlineTopLevel = outlineTopLevel;
                }
            }
            catch (Exception e)
            {
                AnsiConsole.WriteException(e);
            }

            if (!hasAllData)
            {
                statelessInferenceParams = GetStatelessInferenceParams(OUTLINE_TOP_LEVEL_GRAMMAR, 0.8f);
                numRetries++;
            }
        }

        currStatus.Status = InferenceStatus.WriteOutlineTopLevel;

        AnsiConsole.WriteLine($"-->WriteOutlineTopLevel():  {(hasAllData ? "SUCCESS" : "FAILURE")}");
        return hasAllData;
    }

    private static InferenceParams GetStatelessInferenceParams(string grammar, float temperatureOverride = DEFAULT_TEMPERATURE)
    {
        var handle = GrammarParser.Instance.ParseGrammar(grammar);
        var statelessInferenceParams = new InferenceParams()
        {
            SamplingPipeline = new DefaultSamplingPipeline
            {
                Temperature = DEFAULT_TEMPERATURE,
                TopK = DEFAULT_TOP_K,
                TopP = DEFAULT_TOP_P,
                MinP = DEFAULT_MIN_P,
                RepeatPenalty = DEFAULT_REPEAT_PENALTY,
                PenalizeNewline = false,
                Grammar = handle
            },

            AntiPrompts = new List<string> { "Question:", "#", "Question: ", ".\n" },
            MaxTokens = 300
        };
        return statelessInferenceParams;
    }

    private static async Task<bool> WriteOutlineMainPoint(InferenceResultData currStatus, string concatenatedResponse, int mainPointIndex)
    {
        var prompt = $"{SYSTEM_PROMPT} Write your reponse using the following **JSON Format**\n{OUTLINE_MAIN_POINT_JSON}\n\n**CONTEXT**\n{concatenatedResponse}\n\n**MAIN POINT**\n{currStatus.OutlineTopLevel.MainPoints[mainPointIndex]}\n\n**USER**\nCarefully and closely read **CONTEXT** and **MAIN POINT** and Write 3 Supporting Points for it. Then, write a **MAIN POINT SUMMARY**.\n\n**ASSISTANT**\n";

        var statelessInferenceParams = GetStatelessInferenceParams(OUTLINE_MAIN_POINT_GRAMMAR);

        var numRetries = 0;
        var hasAllData = false;
        while ((numRetries < MAX_NUM_RETRIES) && !hasAllData)
        {
            AnsiConsole.WriteLine($"WriteOutlineMainPoint():  MainPoint {mainPointIndex + 1}:  Try {numRetries + 1} of {MAX_NUM_RETRIES}");
            var response = await statelessExecutor.InferAsync(prompt, statelessInferenceParams).ToListAsync();
            currStatus.OutlineResponse = string.Concat(response);

            try
            {
                var outlineMainPoint = JsonSerializer.Deserialize<OutlineMainPoint>(currStatus.OutlineResponse);
                if ((outlineMainPoint != null) && (hasAllData = outlineMainPoint.HasAllData()))
                {
                    currStatus.OutlineMainPoint.Add(mainPointIndex, outlineMainPoint);
                }
            }
            catch (Exception e)
            {
                AnsiConsole.WriteException(e);
            }

            if (!hasAllData)
            {
                statelessInferenceParams = GetStatelessInferenceParams(OUTLINE_MAIN_POINT_GRAMMAR, 0.8f);
                numRetries++;
            }
        }

        switch (mainPointIndex)
        {
            case 0:
                currStatus.Status = InferenceStatus.WriteOutlineMainPoint1;
                break;
            case 1:
                currStatus.Status = InferenceStatus.WriteOutlineMainPoint2;
                break;
            case 2:
                currStatus.Status = InferenceStatus.WriteOutlineMainPoint3;
                break;
            default:
                AnsiConsole.WriteLine($"mainPointIndex out of rangle: {mainPointIndex}");
                break;
        }
        AnsiConsole.WriteLine($"-->WriteOutlineMainPoint():  MainPoint {mainPointIndex + 1}:  {(hasAllData ? "SUCCESS" : "FAILURE")}");
        return hasAllData;
    }

    private static async Task WriteFirstDraft(InferenceResultData currStatus, string concatenatedResponse, string outline)
    {
        //var handle = GrammarParser.Instance.ParseGrammar(OUTLINE_GRAMMAR);
        var statelessInferenceParams = new InferenceParams()
        {
            SamplingPipeline = new DefaultSamplingPipeline
            {
                Temperature = DEFAULT_TEMPERATURE,
                TopK = DEFAULT_TOP_K,
                TopP = DEFAULT_TOP_P,
                MinP = DEFAULT_MIN_P,
                RepeatPenalty = DEFAULT_REPEAT_PENALTY,
                //Grammar = handle
            },

            AntiPrompts = new List<string> { "Question:", "#", "Question: ", ".\n" },
            MaxTokens = 300
        };

        var prompt = $"{SYSTEM_PROMPT}\n\n**OUTLINE**\n{outline}\n\n**USER**\n{currStatus.UserMsg}. Carefully and closely read the **OUTLINE**. Write the paragraph for the first topic. Write at the college-graduate level.\n\n**ASSISTANT**\n";
        var response = await statelessExecutor.InferAsync(prompt, statelessInferenceParams).ToListAsync();
        currStatus.FirstDraft = string.Concat(response);
        currStatus.Status = InferenceStatus.WriteFirstDraft;
    }


    private static void ReportResults()
    {
        AnsiConsole.WriteLine($"TotalBatchTokens: {BatchTokenCount}");
        foreach (var kvp in mapConversationIdToInferenceData)
        {
            AnsiConsole.WriteLine($"Conversation: {kvp.Key}: {kvp.Value.Status}, {kvp.Value.NumTokens} tokens, {kvp.Value.KvCacheResult}");
        }

        for (var conversationIndex = 0; conversationIndex < mapConversationIdToInferenceData.Count; conversationIndex++)
        {
            var currStatus = mapConversationIdToInferenceData[conversationIndex];
            var currGuidedDecoder = currStatus.StreamingTokenDecoder; // guidedDecoder[conversationIndex];
            var msg = currGuidedDecoder.Read().ReplaceLineEndings(" ");

            // Clip the last word.
            if (!string.IsNullOrEmpty(msg) && !msg.EndsWith(' '))
            {
                var lastSpaceIndex = msg.LastIndexOf(" ");
                msg = msg.Substring(0, lastSpaceIndex);
            }
            var lines = msg.Split("\n").ToList();
            for(var index = 0; index < lines.Count; index++)
            {
                lines[index] = lines[index].Trim();
            }
            if (currStatus.Response.Count == 0)
            {
                currStatus.Response.AddRange(lines);
            }
            else
            {
                currStatus.Response.InsertRange(currStatus.Response.Count, lines);
            }

            AnsiConsole.MarkupLine($"[green]Guided: {conversationIndex}: {msg.Length} chars, {currStatus.NumTokens} tokens[/]");
            AnsiConsole.WriteLine($"{msg}");
        }

        var kvNumTokensInCache = executor.Context.NativeHandle.KvCacheCountTokens();
        var kvCacheCellsInUse = executor.Context.NativeHandle.KvCacheCountCells();
        var kvCachePercentInUse = kvNumTokensInCache > 0 ? (float)kvCacheCellsInUse / (float)kvNumTokensInCache : 0.0f;
        AnsiConsole.WriteLine($"kvCache (BEFORE): {kvCachePercentInUse.ToString("N2")}% in use: {kvCacheCellsInUse} / {kvNumTokensInCache}");
        //executor.Context.NativeHandle.KvCacheClear();
        executor.Context.NativeHandle.KvCacheDefrag();
        executor.Context.NativeHandle.KvCacheUpdate();

        kvCacheCellsInUse = executor.Context.NativeHandle.KvCacheCountCells();
        kvCachePercentInUse = kvNumTokensInCache > 0 ? (float)kvCacheCellsInUse / (float)kvNumTokensInCache : 0.0f;
        AnsiConsole.WriteLine($"kvCache (AFTER): {kvCachePercentInUse.ToString("N2")}% in use: {kvCacheCellsInUse} / {kvNumTokensInCache}");

        // Print some stats
        var timings = executor.Context.NativeHandle.GetTimings();
        AnsiConsole.WriteLine($"Total Tokens Evaluated: {timings.TokensEvaluated}");
        AnsiConsole.WriteLine($"Total Tokens Sampled: {timings.TokensSampled}");
        AnsiConsole.WriteLine($"Eval Time: {(timings.Eval + timings.PromptEval).TotalMilliseconds}ms");
        AnsiConsole.WriteLine($"Sample Time: {timings.Sampling.TotalMilliseconds}ms");

        var totalSeconds = timer.Elapsed.TotalSeconds;
        var tokensPerSec = BatchTokenCount / totalSeconds;

        AnsiConsole.MarkupLine($"{tokensPerSec.ToString("N2")} tokens/sec: BatchTokenCount={BatchTokenCount}, elapsed={(timer.Elapsed.TotalSeconds.ToString("N2"))}, InferenceDecodeErrorFound={InferenceDecodeErrorFound}, TotalBatchTokensReached={TotalBatchTokensReached}");

        AnsiConsole.WriteLine("ReportResults(): END.");
        //executor.Context.LoadState("SaveState");
    }

    private static async Task RunBatchedInference()
    {
        for (var tokenIndex = 0; (tokenIndex < MAX_CONVERSATION_TOKEN_COUNT) && !TotalBatchTokensReached && (errors.Count == 0); tokenIndex++)
        {
            for (var conversationIndex = 0; (conversationIndex < NUM_BATCHED_CONVERSATIONS) && !InferenceDecodeErrorFound && !TotalBatchTokensReached; conversationIndex++)
            {
                try
                {
                    var currStatus = mapConversationIdToInferenceData[conversationIndex];
                    if (currStatus.Status != InferenceStatus.GatherNotes)
                    {
                        continue;
                    }

                    var currGuided = currStatus.Conversation;
                    var currGuidedSampler = currStatus.SamplingPipeline as DefaultSamplingPipeline;
                    var currGuidedDecoder = currStatus.StreamingTokenDecoder;

                    if ((currGuided == null) || (currGuidedSampler == null) || (currGuidedDecoder == null))
                    {
                        errors.Add($"currGuided={currGuided}, currGuidedSampler={currGuidedSampler}, currGuidedDecoder={currGuidedDecoder}");
                        break;
                    }

                    // Try to infer on the current conversation. If an error occurs (likely NoKvSlot), set the current
                    // conversations error status and code and abort.
                    if (currGuided.RequiresInference)
                    {
                        decodeResult = await executor.Infer();
                        if (decodeResult != DecodeResult.Ok)
                        {
                            currStatus.KvCacheResult = decodeResult;
                            currStatus.Status = InferenceStatus.Error;
                            errors.Add($"Can't infer on conversation: id={currStatus.Id}");

                            AnsiConsole.WriteLine($"Can't Infer: decodeResult={decodeResult}");
                            break;
                        }
                    }

                    // Sample from the conversation.
                    var currGuidedToken = currGuidedSampler.Sample(executor.Context.NativeHandle, currGuided.Sample(), []);
                    currGuidedDecoder.Add(currGuidedToken);     // Note: token is decoded and added to a list of characters.
                    currGuided.Prompt(currGuidedToken);

                    currStatus.NumTokens++;
                    BatchTokenCount++;

                    // Early exit if we reach the natural end of the response.
                    if (model.Tokens.IsEndOfGeneration(currGuidedToken))
                    {
                        currStatus.Status = InferenceStatus.NotesComplete;
                        AnsiConsole.WriteLine($"EndOfGeneration Reached: tokenIndex={tokenIndex}, conversationIndex={conversationIndex}: {currStatus.NumTokens} tokens");
                    }

                    if (currStatus.NumTokens >= MAX_CONVERSATION_TOKEN_COUNT)
                    {
                        currStatus.Status = InferenceStatus.MaxConversationTokensReached;
                        AnsiConsole.WriteLine($"MaxConversationTokensReached: tokenIndex={tokenIndex}, conversationIndex={conversationIndex}: {currStatus.NumTokens} tokens");
                    }
                }
                catch (Exception e)
                {
                    AnsiConsole.WriteLine($"EXCEPTION: tokenIndex={tokenIndex}, conversationIndex={conversationIndex}: {e.Message}");
                    mapConversationIdToInferenceData[conversationIndex].Status = InferenceStatus.Error;
                }
                finally
                {
                }
            }

            // Update progress bar
            reporter.Increment(1.0);
        }
        timer.Stop();
    }

    #region GuidedSampler
    private class GuidedSampler(Conversation guidance, float weight)
        : BaseSamplingPipeline
    {
        protected override LLamaToken ProcessTokenDataArray(SafeLLamaContextHandle ctx, LLamaTokenDataArray candidates, ReadOnlySpan<LLamaToken> lastTokens)
        {
            // Get the logits generated by the guidance sequences
            var guidanceLogits = guidance.Sample();

            // Modify these logits based on the guidance logits
            candidates.Guidance(ctx, guidanceLogits, weight);

            // Basic sampling
            candidates.Temperature(ctx, 0.8f);
            candidates.TopK(ctx, 25);
            return candidates.SampleToken(ctx);
        }
        
        public override void Accept(SafeLLamaContextHandle ctx, LLamaToken token)
        {
        }
        
        public override ISamplingPipeline Clone()
        {
            throw new NotSupportedException();
        }
        
        protected override void ProcessLogits(SafeLLamaContextHandle ctx, Span<float> logits, ReadOnlySpan<LLamaToken> lastTokens)
        {
        }
    }
    #endregion

    #region Test Prompts
    private static readonly string[] questions =
    [
        "What are the main ingredients in a classic Margherita pizza?",
        "Can you explain the water cycle in simple terms?",
        "Write a short story about a cat who discovers a hidden treasure.",
        "Compose a poem about the changing seasons.",
        "How would you approach organizing a community event?",
        "What are some strategies for managing time effectively during a busy week?",
        "What are three interesting facts about dolphins?",
        "Tell me about the origin of the word \"quarantine.\"",
        "Who was Marie Curie and what were her contributions to science?",
        "Describe the significance of the Great Wall of China.",
        "What are some tips for maintaining a healthy work-life balance?",
        "How can I create a budget for my monthly expenses?",
        "Explain the concept of photosynthesis in plants.",
        "What are the different phases of the moon?",
        "What are the benefits of using cloud storage for personal files?",
        "How does a basic search engine work?",
        "What are the characteristics of Impressionist painting?",
        "Describe the significance of the Mona Lisa in art history.",
        "What are the main themes in Shakespeare's \"Romeo and Juliet?\"",
        "Can you summarize the plot of \"The Great Gatsby\"?",
        "What are some popular tourist attractions in Paris?",
        "Describe the landscape and culture of the Amazon rainforest.",
        "What are the benefits of regular physical exercise?",
        "How can mindfulness meditation improve mental health?",
        "What are some easy recipes for a beginner cook?",
        "Explain the difference between baking and roasting.",
        "What are the main causes of climate change?",
        "How can individuals reduce their carbon footprint?",
        "What are the basic rules of soccer?",
        "Describe the benefits of participating in team sports.",
        "What are some effective ways to improve public speaking skills?",
        "Explain the importance of learning a second language.",
        "What are some effective strategies for setting and achieving personal goals?",
        "How can journaling benefit mental well-being?",
        "What are the key characteristics of a rainforest ecosystem?",
        "Describe the life cycle of a butterfly.",
        "What were the main causes of World War I?",
        "Who was Nelson Mandela and what was his impact on South Africa?",
        "What are the advantages of using renewable energy sources?",
        "How has the internet changed the way we communicate?",
        "What are the different genres of music and how do they differ?",
        "Can you explain the influence of jazz on modern music?",
        "What are some effective study techniques for students?",
        "How can technology enhance the learning experience in classrooms?",
        "What are some popular hobbies that people enjoy in their free time?",
        "How can gardening be a rewarding activity?",
        "What are the benefits of volunteering in your community?",
        "How can social media positively impact community engagement?",
        "What is the basic principle behind how a refrigerator works?",
        "Explain the difference between renewable and non-renewable resources.",
        "What are some common signs of stress and how can they be managed?",
        "How does positive reinforcement work in behavior modification?",
        "What are some timeless fashion trends that never go out of style?",
        "How can someone develop their own personal style?",
        "What are some effective strategies for positive discipline in children?",
        "How can families create a strong bond through shared activities?",
        "What are the basics of investing in the stock market?",
        "Explain the importance of saving for retirement.",
        "What is the concept of utilitarianism in ethics?",
        "How can critical thinking improve decision-making?",
        "What are the benefits of yoga for physical and mental health?",
        "How can someone start a running routine as a beginner?",
        "What are the potential benefits of artificial intelligence in everyday life?",
        "How has telecommuting changed the workplace dynamic?",
        "What are the health benefits of a balanced diet?",
        "Can you suggest some nutritious snacks for kids?",
        "What are some essential items to pack for a weekend getaway?",
        "How can travelers stay safe while exploring a new city?",
        "What are some simple ways to improve indoor air quality?",
        "How can someone start a small vegetable garden at home?",
        "What are the key features to look for in a smartphone?",
        "How can smart home devices enhance daily living?",
        "What are some beginner-friendly DIY craft projects?",
        "How can painting be a therapeutic activity?",
        "What are some unique traditions celebrated around the world?",
        "How has the role of women in society changed over the past century?",
        "What are some tips for creating a successful savings plan?",
        "How can someone improve their credit score?",
        "What are the benefits of using educational apps for learning?",
        "How can online courses provide flexible learning opportunities?",
        "What are some everyday practices to reduce plastic waste?",
        "How does recycling benefit the environment?",
        "What are effective ways to resolve conflicts in a relationship?",
        "How can active listening improve communication skills?",
        "What are the pros and cons of social media in today's world?",
        "How has technology influenced the way we shop?",
        "What are the benefits of reading regularly?",
        "Can you recommend some classic novels for beginners?",
        "What are some low-impact exercises for seniors?",
        "How can someone stay motivated to maintain a fitness routine?",
        "What are some habits that can lead to personal success?",
        "How can practicing gratitude improve overall happiness?",
        "What are the key elements of a good movie?",
        "How can attending live performances enhance the appreciation of art?",
        "What are some fascinating discoveries made in space exploration?",
        "How do scientists study climate change?",
        "What are some common symptoms of dehydration and how can it be prevented?",
        "How does sleep affect overall health and well-being?",
        "What are the potential benefits of 3D printing in manufacturing?",
        "How can virtual reality be used in education?",
        "What are some key aspects of Japanese culture that are unique?",
        "How do festivals contribute to cultural identity?",
        "What are the benefits of planting native species in gardens?",
        "How can someone attract pollinators to their garden?",
        "What are some tips for managing student loan debt?",
        "How can someone start investing with a small budget?",
        "What are some fun and educational activities for toddlers?",
        "How can parents encourage a love of reading in their children?",
        "What were the major achievements of the Renaissance period?",
        "How has technology changed the way we access information?",
        "What are some tips for traveling on a budget?",
        "How can solo travel be a rewarding experience?",
        "What are some tips for meal prepping for the week?",
        "How can someone make a healthy smoothie?",
        "What are the benefits of using a password manager?",
        "How can digital detoxing improve mental health?",
        "What are some easy sewing projects for beginners?",
        "How can upcycling materials create unique home decor?",
        "What are the differences between mammals and reptiles?",
        "How does the process of photosynthesis benefit the environment?",
        "What are some effective techniques for building self-confidence?",
        "How can someone develop a growth mindset?",
        "What are the benefits of participating in outdoor activities?",
        "How can someone get started with a new sport?",
        "What are some significant landmarks in ancient Egypt?",
        "How do natural disasters impact communities?",
        "What are the benefits of community service for individuals?",
        "How can local businesses support community development?",
        "What are some foods that are high in antioxidants?",
        "How can someone incorporate more fruits and vegetables into their diet?",
        "What are some emerging technologies that could change our daily lives?",
        "How might artificial intelligence impact the job market in the next decade?",
        "What are some tips for traveling with pets?",
        "How can someone choose the best travel insurance for their needs?",
        "What are the benefits of creating an emergency fund?",
        "How can someone start a side hustle to earn extra income?",
        "What are some common elements of a compelling story?",
        "How can someone improve their creative writing skills?",
        "What are the benefits of participating in local clean-up events?",
        "How can individuals contribute to wildlife conservation efforts?",
        "What are some influential movements in modern art?",
        "How does music influence cultural identity?",
        "What are some strategies for helping children with homework?",
        "How can parents foster creativity in their children?",
        "What are some easy recipes for quick weeknight dinners?",
        "How can someone make a delicious homemade pizza?",
        "What are the advantages of using a smartwatch?",
        "How can someone protect their privacy online?",
        "What are some effective methods for overcoming procrastination?",
        "How can setting daily intentions improve productivity?",
        "What are the benefits of birdwatching as a hobby?",
        "How can urban areas create more green spaces?",
        "What are some key events that shaped the Civil Rights Movement?",
        "How do cultural exchanges benefit societies?",
        "What are some effective stretches for improving flexibility?",
        "How can someone stay active during the winter months?",
        "What are some fun and easy painting techniques for beginners?",
        "How can scrapbooking be a creative outlet?",
        "What are some must-see destinations for nature lovers?",
        "How can someone travel sustainably and minimize their impact?",
        "What are some simple ways to incorporate mindfulness into daily life?",
        "How can someone create a balanced morning routine?",
        "What are the benefits of using cloud computing for businesses?",
        "How can augmented reality enhance shopping experiences?",
        "What are some tips for starting a compost bin at home?",
        "How can container gardening be a solution for small spaces?",
        "What are some strategies for reducing monthly expenses?",
        "How can someone effectively track their spending habits?",
        "What are some characteristics of a strong protagonist in a story?",
        "How can someone develop a unique writing voice?",
        "What are the benefits of participating in local advocacy groups?",
        "How can community gardens promote social interaction?",
        "What are some classic films that everyone should watch?",
        "How can attending art exhibitions enhance appreciation for creativity?",
        "What are some tips for solo travelers to stay safe?",
        "How can someone find hidden gems while traveling?",
        "What are some healthy alternatives to common snacks?",
        "How can someone learn to cook if they are a complete beginner?",
        "What are the benefits of using open-source software?",
        "How has online education changed the landscape of learning?",
        "What are some techniques for effective goal setting?",
        "How can someone develop better time management skills?",
        "What are the key benefits of planting trees in urban areas?",
        "How can individuals participate in local conservation efforts?",
        "What were the main causes and effects of the Industrial Revolution?",
        "How do historical monuments contribute to cultural identity?",
        "What are some fun ways to stay active with friends?",
        "How can someone create a balanced workout routine?",
        "What are some beginner-friendly knitting projects?",
        "How can someone use photography to express their creativity?",
        "What are some cultural etiquette tips for travelers?",
        "How can exploring local cuisine enhance travel experiences?",
        "What are some easy meal ideas for a vegetarian diet?",
        "How can someone make healthy choices when dining out?",
        "What are the advantages of using video calls for remote work?",
        "How can emojis enhance digital communication?",
        "What are some strategies for building resilience in tough times?",
        "How can someone practice self-compassion?",
        "What are the benefits of creating a wildlife-friendly garden?",
        "How can people get involved in local wildlife conservation efforts?",
        "What were the key events that led to the fall of the Roman Empire?",
        "How do cultural festivals celebrate diversity in communities?",
        "What are some effective home workouts for beginners?",
        "How can someone stay motivated to exercise regularly?",
        "What are some popular genres of music and their characteristics?",
        "How can attending theater performances enrich cultural understanding?",
        "What are some tips for planning a road trip?",
        "How can someone find volunteer opportunities while traveling?",
        "What are some popular international dishes to try at home?",
        "How can someone create a balanced meal plan for the week?",
        "What are the benefits of using electric vehicles?",
        "How does blockchain technology work and what are its potential uses?",
        "What are some effective ways to practice gratitude daily?",
        "How can someone improve their public speaking skills?",
        "What are the benefits of using native plants in landscaping?",
        "How can individuals reduce water consumption at home?",
        "What are some significant contributions of ancient Greece to modern society?",
        "How do traditional crafts reflect cultural heritage?",
        "What are some benefits of practicing yoga?",
        "How can someone incorporate more physical activity into a busy schedule?",
        "What are some beginner painting techniques to explore?",
        "How can someone start a scrapbook to preserve memories?",
        "What are some tips for packing efficiently for a trip?",
        "How can someone experience local culture while traveling?",
        "What are some foods that are high in antioxidants?",
        "How can someone incorporate more fruits and vegetables into their diet?",
        "What are some emerging technologies that could change our daily lives?",
        "How might artificial intelligence impact the job market in the next decade?",
        "What are some tips for traveling with pets?",
        "How can someone choose the best travel insurance for their needs?",
        "What are the benefits of creating an emergency fund?",
        "How can someone start a side hustle to earn extra income?",
        "What are some common elements of a compelling story?",
        "How can someone improve their creative writing skills?",
        "What are the benefits of participating in local clean-up events?",
        "How can individuals contribute to wildlife conservation efforts?",
        "What are some influential movements in modern art?",
        "How does music influence cultural identity?",
        "What are some strategies for helping children with homework?",
        "How can parents foster creativity in their children?",
        "What are some easy recipes for quick weeknight dinners?",
        "How can someone make a delicious homemade pizza?",
        "What are the advantages of using a smartwatch?",
        "How can someone protect their privacy online?",
        "What are some effective methods for overcoming procrastination?",
        "How can setting daily intentions improve productivity?",
        "What are the benefits of birdwatching as a hobby?",
        "How can urban areas create more green spaces?",
        "What are some key events that shaped the Civil Rights Movement?",
        "How do cultural exchanges benefit societies?",
        "What are some effective stretches for improving flexibility?",
        "How can someone stay active during the winter months?",
        "What are some fun and easy painting techniques for beginners?",
        "How can scrapbooking be a creative outlet?",
        "What are some must-see destinations for nature lovers?",
        "How can someone travel sustainably and minimize their impact?",
        "What are some simple ways to incorporate mindfulness into daily life?",
        "How can someone create a balanced morning routine?",
        "What are the benefits of using cloud computing for businesses?",
        "How can augmented reality enhance shopping experiences?",
        "What are some tips for starting a compost bin at home?",
        "How can container gardening be a solution for small spaces?",
        "What are some strategies for reducing monthly expenses?",
        "How can someone effectively track their spending habits?",
        "What are some characteristics of a strong protagonist in a story?",
        "How can someone develop a unique writing voice?",
        "What are the benefits of participating in local advocacy groups?",
        "How can community gardens promote social interaction?",
        "What are some classic films that everyone should watch?",
        "How can attending art exhibitions enhance appreciation for creativity?",
        "What are some tips for solo travelers to stay safe?",
        "How can someone find hidden gems while traveling?",
        "What are some healthy alternatives to common snacks?",
        "How can someone learn to cook if they are a complete beginner?",
        "What are the benefits of using open-source software?",
        "How has online education changed the landscape of learning?",
        "What are some techniques for effective goal setting?",
        "How can someone develop better time management skills?",
        "What are the key benefits of planting trees in urban areas?",
        "How can individuals participate in local conservation efforts?",
        "What were the main causes and effects of the Industrial Revolution?",
        "How do historical monuments contribute to cultural identity?",
        "What are some fun ways to stay active with friends?",
        "How can someone create a balanced workout routine?",
        "What are some beginner-friendly knitting projects?",
        "How can someone use photography to express their creativity?",
        "What are some cultural etiquette tips for travelers?",
        "How can exploring local cuisine enhance travel experiences?",
        "What are some easy meal ideas for a vegetarian diet?",
        "How can someone make healthy choices when dining out?",
        "What are the advantages of using video calls for remote work?",
        "How can emojis enhance digital communication?",
        "What are some strategies for building resilience in tough times?",
        "How can someone practice self-compassion?",
        "What are the benefits of creating a wildlife-friendly garden?",
        "How can people get involved in local wildlife conservation efforts?",
        "What were the key events that led to the fall of the Roman Empire?",
        "How do cultural festivals celebrate diversity in communities?",
        "What are some effective home workouts for beginners?",
        "How can someone stay motivated to exercise regularly?",
        "What are some popular genres of music and their characteristics?",
        "How can attending theater performances enrich cultural understanding?",
        "What are some tips for planning a road trip?",
        "How can someone find volunteer opportunities while traveling?",
        "What are some popular international dishes to try at home?",
        "How can someone create a balanced meal plan for the week?",
        "What are the benefits of using electric vehicles?",
        "How does blockchain technology work and what are its potential uses?",
        "What are some effective ways to practice gratitude daily?",
        "How can someone improve their public speaking skills?",
        "What are the benefits of using native plants in landscaping?",
        "How can individuals reduce water consumption at home?",
        "What are some significant contributions of ancient Greece to modern society?",
        "How do traditional crafts reflect cultural heritage?",
        "What are some benefits of practicing yoga?",
        "How can someone incorporate more physical activity into a busy schedule?",
        "What are some beginner painting techniques to explore?",
        "How can someone start a scrapbook to preserve memories?",
        "What are some tips for packing efficiently for a trip?",
        "How can someone experience local culture while traveling?",
        "What are some tips for baking the perfect loaf of bread at home?",
        "How can someone make a flavorful homemade pasta sauce?",
        "What are the benefits of using smart home devices for energy efficiency?",
        "How can someone set up a secure home Wi-Fi network?",
        "What are some daily habits that can improve mental well-being?",
        "How can practicing mindfulness help reduce anxiety?",
        "What are the ecological benefits of wetlands?",
        "How can individuals support endangered species conservation?",
        "What were the major achievements of the Ancient Egyptians?",
        "How do museums contribute to the preservation of history?",
        "What are some effective ways to incorporate more movement into a sedentary lifestyle?",
        "How can someone start a meditation practice?",
        "What are some creative ways to repurpose old clothing?",
        "How can someone get started with digital illustration?",
        "What are some tips for experiencing local culture in a foreign country?",
        "How can someone plan a hiking trip in a national park?",
        "What are some quick and healthy lunch ideas for work or school?",
        "How can someone make a delicious and nutritious smoothie bowl?",
        "What are the benefits of using a VPN for online privacy?",
        "How can artificial intelligence enhance customer service?",
        "What are some effective techniques for improving focus and concentration?",
        "How can someone develop a positive mindset?",
        "What are the benefits of using public transportation for the environment?",
        "How can individuals reduce their carbon footprint at home?",
        "What were the main causes of the American Revolution?",
        "How do cultural traditions influence modern celebrations?",
        "What are some fun ways to stay active during the summer?",
        "How can someone create a balanced meal plan for weight loss?",
        "What are some beginner-friendly photography tips for capturing great images?",
        "How can someone start a bullet journal to organize their thoughts and tasks?",
        "What are some tips for exploring a new city on foot?",
        "How can someone find affordable accommodations while traveling?",
    ];
    #endregion
}
