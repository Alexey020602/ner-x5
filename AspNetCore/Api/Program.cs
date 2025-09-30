using System.Text.Json;
using CSnakes.Runtime;
using DotNetEnv;
using DotNetEnv.Configuration;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);
if (builder.Environment.IsDevelopment())
{
    builder.Configuration.AddDotNetEnv("../../ML_Part/integration/.env", LoadOptions.TraversePath());
}

// builder.Services.AddLogging();
builder.Logging.AddFilter("CSnakes", LogLevel.Information);
builder.Services.AddOpenApi();

var modelSection = builder.Configuration.GetSection("HuggingFace");
var modelRepoName = modelSection.GetValue<string>("ModelRepoName") ?? throw new InvalidOperationException("Configuration must contain model repo name.");
var modelRepoToken = modelSection.GetValue<string>("ModelRepoToken")  ?? throw new InvalidOperationException("Configuration must contain model repo token.");

string home;

if (builder.Environment.IsDevelopment())
{
    home = Path.Join(
        builder.Environment.ContentRootPath,
        "..", "..", 
        "ML_PART", 
        "integration");
} 
else
{
    home = Path.Join(builder.Environment.ContentRootPath, "python");
}
var venv = Path.Join(home, "venv");
builder.Services.WithPython()
    .WithHome(home)
    .WithVirtualEnvironment(venv)
    .WithPipInstaller()
    .FromRedistributable("3.12")
    ;

builder.Services.AddSingleton(sp => sp.GetRequiredService<IPythonEnvironment>().Model());

builder.AddServiceDefaults();

builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
    options.SerializerOptions.PropertyNameCaseInsensitive = true;
});

var app = builder.Build();

using (var scope = app.Services.CreateScope())
{
    var logger = scope.ServiceProvider.GetRequiredService<ILogger<Program>>();
    logger.LogInformation("Python home path {Home}", home);
    logger.LogInformation("Current Directory: {CurrentDirectory}", Environment.CurrentDirectory);
    // logger.LogInformation("wwwroot path {WwwRoot}", app.Environment.WebRootPath);
    logger.LogInformation("current root path {CurrentRoot}", app.Environment.ContentRootPath);
    logger.LogInformation("venv path {Venv}", venv);
    logger.LogInformation("Model repo name {ModelPath}", modelRepoName);
    try
    {
        scope.ServiceProvider.GetRequiredService<IModel>().Initialize(modelRepoName, modelRepoToken);
    }
    catch (PythonInvocationException ex)
    {
        logger.LogError(ex, "Python invocation exception");
        if (ex.InnerException is PythonRuntimeException pythonRuntimeException)
        {
            logger.LogError("{Message}", pythonRuntimeException.Message);
            foreach (var stackTraceItem in pythonRuntimeException.PythonStackTrace)
            {
                logger.LogInformation(stackTraceItem);
            }
        }
        throw;
    }
} 

app.MapDefaultEndpoints();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
    app.MapScalarApiReference();
}

app.UseHttpsRedirection();

app.MapPost("/api/predict", 
    (PredictionRequest request, IModel module) =>
    {
        var result = module.Predict(request.Input);
        return result.Select(tuple => new PredictionItem(tuple.Item1, tuple.Item2, tuple.Item3));
    });

await app.RunAsync();
return;