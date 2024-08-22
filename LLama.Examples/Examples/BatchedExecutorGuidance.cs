using LLama.Batched;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using Spectre.Console;

namespace LLama.Examples.Examples;

/// <summary>
/// This demonstrates using a batch to generate two sequences and then using one
/// sequence as the negative guidance ("classifier free guidance") for the other.
/// </summary>
public class BatchedExecutorGuidance
{
    /// <summary>
    /// Set how many tokens should be generated
    /// </summary>
    private const int TokenCount = 32;

    public static async Task Run()
    {
        // Load model weights
        var parameters = new ModelParams(UserSettings.GetModelPath());
        using var model = await LLamaWeights.LoadFromFileAsync(parameters);

        var positivePrompt = AnsiConsole.Ask("Positive Prompt (or ENTER for default):", "My favourite colour is").Trim();
        var negativePrompt = AnsiConsole.Ask("Negative Prompt (or ENTER for default):", "I hate the colour red. My favourite colour is").Trim();
        var weight = AnsiConsole.Ask("Guidance Weight (or ENTER for default):", 2.0f);

        // Create an executor that can evaluate a batch of conversations together
        using var executor = new BatchedExecutor(model, parameters);

        // Print some info
        var name = model.Metadata.GetValueOrDefault("general.name", "unknown model name");
        Console.WriteLine($"Created executor with model: {name}");

        // Load the two prompts into two conversations
        using var guided = executor.Create();
        guided.Prompt(executor.Context.Tokenize(positivePrompt));
        using var guidance = executor.Create();
        guidance.Prompt(executor.Context.Tokenize(negativePrompt));

        // Run inference to evaluate prompts
        await AnsiConsole
             .Status()
             .Spinner(Spinner.Known.Line)
             .StartAsync("Evaluating Prompts...", _ => executor.Infer());

        // Fork the "guided" conversation. We'll run this one without guidance for comparison
        using var unguided = guided.Fork();

        var handle = GrammarParser.Instance.ParseGrammar("root ::= \"{\" ws01 root-answer \"}\" ws01\r\nroot-answer ::= \"\\\"answer\\\"\" \":\" ws01 string\r\n\r\n\r\nvalue  ::= (object | array | string | number | boolean | null) ws\r\n\r\nobject ::=\r\n  \"{\" ws (\r\n    string \":\" ws value\r\n    (\",\" ws string \":\" ws value)*\r\n  )? \"}\"\r\n\r\narray  ::=\r\n  \"[\" ws01 (\r\n            value\r\n    (\",\" ws01 value)*\r\n  )? \"]\"\r\n\r\nstring ::=\r\n  \"\\\"\" (string-char)* \"\\\"\"\r\n\r\nstring-char ::= [^\"\\\\] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes\r\n\r\nnumber ::= integer (\".\" [0-9]+)? ([eE] [-+]? [0-9]+)?\r\ninteger ::= \"-\"? ([0-9] | [1-9] [0-9]*)\r\nboolean ::= \"true\" | \"false\"\r\nnull ::= \"null\"\r\n\r\n# Optional space: by convention, applied in this grammar after literal chars when allowed\r\nws ::= ([ \\t\\n] ws)?\r\nws01 ::= ([ \\t\\n])?");

        // Run inference loop
        var unguidedSampler = new DefaultSamplingPipeline();
        var unguidedDecoder = new StreamingTokenDecoder(executor.Context);
        var guidedSampler = new GuidedSampler(guidance, weight);
        unguidedSampler.Grammar = handle;
        guidedSampler.Grammar = handle;
        var guidedDecoder = new StreamingTokenDecoder(executor.Context);
        await AnsiConsole
           .Progress()
           .StartAsync(async progress =>
            {
                var reporter = progress.AddTask("Running Inference", maxValue: TokenCount);

                for (var i = 0; i < TokenCount; i++)
                {
                    if (i != 0)
                        await executor.Infer();

                    // Sample from the "unguided" conversation. This is just a conversation using the same prompt, without any
                    // guidance. This serves as a comparison to show the effect of guidance.
                    var u = unguidedSampler.Sample(executor.Context.NativeHandle, unguided.Sample(), []);
                    unguidedDecoder.Add(u);
                    unguided.Prompt(u);

                    // Sample from the "guided" conversation. This sampler will internally use the "guidance" conversation
                    // to steer the conversation. See how this is done in GuidedSampler.ProcessLogits (bottom of this file).
                    var g = guidedSampler.Sample(executor.Context.NativeHandle, guided.Sample(), []);
                    guidedSampler.Accept(executor.Context.NativeHandle, g);
                    guidedDecoder.Add(g);

                    // Use this token to advance both guided _and_ guidance. Keeping them in sync (except for the initial prompt).
                    guided.Prompt(g);
                    guidance.Prompt(g);

                    // Early exit if we reach the natural end of the guided sentence
                    if (model.Tokens.IsEndOfGeneration(g))
                        break;

                    // Update progress bar
                    reporter.Increment(1);
                }
            });

        AnsiConsole.MarkupLine($"[green]Unguided:[/][white]{unguidedDecoder.Read().ReplaceLineEndings(" ")}[/]");
        AnsiConsole.MarkupLine($"[green]Guided:[/][white]{guidedDecoder.Read().ReplaceLineEndings(" ")}[/]");
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
        
        public override ISamplingPipeline Clone()
        {
            throw new NotSupportedException();
        }
        
        protected override void ProcessLogits(SafeLLamaContextHandle ctx, Span<float> logits, ReadOnlySpan<LLamaToken> lastTokens)
        {
        }
    }
}