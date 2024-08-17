using LLama.Batched;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using Spectre.Console;
using System.Diagnostics;

namespace LLama.Examples.Examples;

/// <summary>
/// This demonstrates using a batch to generate two sequences and then using one
/// sequence as the negative guidance ("classifier free guidance") for the other.
/// </summary>
public class BatchedExecutorMultiGuidance
{
    /// <summary>
    /// Set how many tokens should be generated
    /// </summary>
    private const int TokenCount = 200;

    private const int NUM_BATCHED_CONVERSATIONS = 64;

    private static async Task<DecodeResult> Infer(BatchedExecutor executor)
    {
        return await executor.Infer();
	}

    public static async Task Run()
    {
        // Load model weights
        var parameters = new ModelParams(UserSettings.GetModelPath());
        //parameters.GpuLayerCount = 8;
        //parameters.NoKqvOffload = true;
        using var model = await LLamaWeights.LoadFromFileAsync(parameters);

        //var positivePrompt = AnsiConsole.Ask("Positive Prompt (or ENTER for default):", "My favourite colour is").Trim();
        //var negativePrompt = AnsiConsole.Ask("Negative Prompt (or ENTER for default):", "I hate the colour red. My favourite colour is").Trim();
        var weight = AnsiConsole.Ask("Guidance Weight (or ENTER for default):", 2.0f);

        // Create an executor that can evaluate a batch of conversations together
        using var executor = new BatchedExecutor(model, parameters);

        executor.Context.SaveState("SaveState");

        // Print some info
        var name = model.Metadata.GetValueOrDefault("general.name", "unknown model name");
        Console.WriteLine($"Created executor with model: {name}");

        // Load the two prompts into two conversations
        var guided = new List<Conversation>();
        var guidedSampler = new List<DefaultSamplingPipeline>();
        var guidedDecoder = new List<StreamingTokenDecoder>();
        var guidedInProgress = new HashSet<int>();
        var guidedNumTokens = new List<int>();

        for (var index = 0; index < NUM_BATCHED_CONVERSATIONS; index++)
        {
            var promptIndex = index % questions.Length;
            var currPrompt = questions[promptIndex];

            var currGuided = executor.Create();
            guided.Add(currGuided);
            currGuided.Prompt(executor.Context.Tokenize(currPrompt));

            //guidedSampler.Add(new GuidedSampler(currGuided, weight));
            guidedSampler.Add(new DefaultSamplingPipeline());
            guidedDecoder.Add(new StreamingTokenDecoder(executor.Context));

            // Initialize the conversations that are active and the number of tokens they've received.
            guidedInProgress.Add(index);
            guidedNumTokens.Add(0);
        }

        var tokenCount = NUM_BATCHED_CONVERSATIONS; // guided and unguided.
        var timer = Stopwatch.StartNew();
        var decodeResult = DecodeResult.Ok;
        var errors = new List<string>();

        // Run inference loop
        await AnsiConsole
           .Progress()
           .StartAsync(async progress =>
            {
                var reporter = progress.AddTask("Running Inference", maxValue: TokenCount);
                for (var tokenIndex = 0; (tokenIndex < TokenCount); tokenIndex++)
                {
                    if (guidedInProgress.Count == 0)
                    {
                        break;
                    }

                    for(var conversationIndex =  0; (conversationIndex < NUM_BATCHED_CONVERSATIONS); conversationIndex++)
                    {
                        try
                        {
                            if (!guidedInProgress.Contains(conversationIndex))
                            {
                                continue;
                            }

                            var currGuided = guided[conversationIndex];
                            var currGuidedSampler = guidedSampler[conversationIndex];
                            var currGuidedDecoder = guidedDecoder[conversationIndex];

                            if (currGuided.RequiresInference)
                            {
                                decodeResult = await executor.Infer();
                                if (decodeResult != DecodeResult.Ok)
                                {
                                    AnsiConsole.WriteLine($"Can't Infer: decodeResult={decodeResult}");
                                    break;
                                }
                            }

                            // Sample from the "guided" conversation. This sampler will internally use the "guidance" conversation
                            // to steer the conversation. See how this is done in GuidedSampler.ProcessLogits (bottom of this file).
                            var currGuidedToken = currGuidedSampler.Sample(executor.Context.NativeHandle, currGuided.Sample(), []);
                            currGuidedDecoder.Add(currGuidedToken);
                            currGuided.Prompt(currGuidedToken);

                            var numTokens = guidedNumTokens[conversationIndex];
                            guidedNumTokens[conversationIndex] = numTokens + 1;
                            tokenCount++;

                            // Early exit if we reach the natural end of the guided sentence
                            if (model.Tokens.IsEndOfGeneration(currGuidedToken))
                            {
                                guidedInProgress.Remove(conversationIndex);
                                AnsiConsole.WriteLine($"EndOfGeneration Reached: tokenIndex={tokenIndex}, conversationIndex={conversationIndex}: {guidedNumTokens[conversationIndex]} tokens");
                            }
                        }
                        catch (Exception e)
                        {
                            AnsiConsole.WriteLine($"EXCEPTION: tokenIndex={tokenIndex}, conversationIndex={conversationIndex}: {e.Message}");
                            guidedInProgress.Remove(conversationIndex);
                        }
                    }

                    if (decodeResult != DecodeResult.Ok)
                    {
                        break;
                    }

                    // Update progress bar
                    reporter.Increment(1.0);
                }
                timer.Stop();
           }
        );

        executor.Context.LoadState("SaveState");

        //AnsiConsole.MarkupLine($"[green]Unguided:[/][white]{unguidedDecoder.Read().ReplaceLineEndings(" ")}[/]");
        for (var conversationIndex = 0; conversationIndex < guidedDecoder.Count; conversationIndex++)
        {
            //AnsiConsole.MarkupLine($"[green]Guided:[/][white]{guidedDecoder.Read().ReplaceLineEndings(" ")}[/]");
            var currGuidedDecoder = guidedDecoder[conversationIndex];
            var msg = currGuidedDecoder.Read().ReplaceLineEndings(" ");
            var numGuidedTokens = guidedNumTokens[conversationIndex];
            AnsiConsole.MarkupLine($"[green]Guided: {conversationIndex}: {msg.Length} chars, {numGuidedTokens} tokens; [/][white]{msg}[/]");
        }

        var kvNumTokensInCache = executor.Context.NativeHandle.KvCacheCountTokens();
        var kvCacheCellsInUse = executor.Context.NativeHandle.KvCacheCountCells();
        var kvCachePercentInUse = (float)kvCacheCellsInUse / (float)kvNumTokensInCache;
        AnsiConsole.WriteLine($"kvCache (BEFORE): {kvCachePercentInUse.ToString("N2")}% in use: {kvCacheCellsInUse} / {kvNumTokensInCache}");
        executor.Context.NativeHandle.KvCacheClear();

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
        var tokensPerSec = tokenCount / totalSeconds;

        AnsiConsole.MarkupLine($"{tokensPerSec.ToString("N2")} tokens/sec: tokenCount={tokenCount}, elapsed={(timer.Elapsed.TotalSeconds.ToString("N2"))}");
    }

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
}
