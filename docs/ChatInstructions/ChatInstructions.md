🚨 CRITICAL: MANDATORY DEVELOPMENT PROTOCOL 🚨
⛔ STOP - READ THESE INSTRUCTIONS COMPLETELY BEFORE ANY ACTION ⛔
PHASE 1: ANALYSIS ONLY (NO CODE GENERATION ALLOWED)
You MUST output the following BEFORE writing ANY code:
A. PROJECT UNDERSTANDING (REQUIRED)
    • CLEARLY STATE what you understand the task to be
    • IDENTIFY all key requirements and constraints
    • ACKNOWLEDGE any ambiguities or uncertainties
B. PLANNED ACTIONS (REQUIRED)
    • LIST each step you will take, numbered and specific
    • EXPLAIN the rationale for your approach
    • SPECIFY which modules, structs, and functions you plan to create
⚠️ CRITICAL STOP POINT ⚠️
After completing Phase 1, you MUST:
    • HALT ALL ACTIVITY
    • AWAIT EXPLICIT APPROVAL with the words "proceed with code generation"
    • DO NOT generate, suggest, or outline any code until given explicit permission

📋 ABSOLUTE REQUIREMENTS FOR ALL CODE GENERATION
THESE RULES ARE NON-NEGOTIABLE:
    1. ✅ JULIA BEST PRACTICES - Strict adherence to established conventions
    2. 🚫 FORBIDDEN: Triple-Quote Comments
        ○ Will cause parsing errors
        ○ Use # for all comments without exception
    3. 🚫 FORBIDDEN: Dictionary-Based Code
        ○ Use direct struct definitions ONLY
        ○ No Dict() for data structures
    4. 📦 MANDATORY: Module Structure
        ○ ALL code must be within properly defined modules
        ○ No loose functions or variables
    5. 📝 MANDATORY: Correct Include/Using Order
        ○ Verify dependency order before output
        ○ Test import sequence validity
    6. 💯 MANDATORY: Complete Files Only
        ○ Output ONLY complete, runnable code files via Canvas
        ○ NO fragments, NO snippets, NO partial implementations
    7. 🛑 MANDATORY: Pre-Code Checkpoint
        ○ ALWAYS present understanding and plan FIRST
        ○ ALWAYS wait for approval BEFORE coding
        ○ ALWAYS ask clarifying questions if needed
    8. 🚫 FORBIDDEN: No new code files
        ○ ALWAYS ask permission before creating a new code file
        ○ ALWAYS state the reason why it's better to create a new file than to use the existing file
    9. 🚫 FORBIDDEN: No sidelining or bypassing test issues
        ○ ALWAYS fix errors as they are encountered
        ○ NEVER modify a test to make it easier to pass. 
        ○ FIX every issue during testing
        ○ Keep a list of bugs that need to be fixed. 
    
❌ FAILURE TO COMPLY = PROJECT REJECTION ❌
CONFIRM: Type "UNDERSTOOD - Awaiting task description" to acknowledge these requirements.

When Responding
Output your understanding in the following structure:
    1. Summary of the project (≤200 words)
    2. Key components and their roles
    3. My interpretation of the task goals
    4. Planned approach / steps I will take
    5. Clarifying questions (if any)
STOP after this step. Do not generate any code until further instructions.

Project Status
    • Core Julia code developed (see uploaded files)
    • Chunk 1 developed and tested
    • Chunk 2 developed and tested
    • Chunk 3 developed and tested
    • Chunk 4 developed, ready for testing

Related Files

Core Files for Testing (REQUIRED)
    1. test_chunk4.jl - The main test file
    2. WeightedPrediction.jl - The merged module (created in this session)
    3. load_all.jl - The fixed module loader (created in this session)
    4. SyntheticSignalGenerator.jl - WITH the patch applied (modified version)
Supporting Files (LIKELY NEEDED)
    1. PredictionMetrics.jl - Performance metrics module
    2. ParameterEncoding.jl - Parameter encoding/decoding
    3. ModernConfigSystem.jl - Configuration system
    4. ProductionFilterBank.jl - Filter implementations
Optional but Helpful Files
    1. run_chunk4_tests.jl - The test runner script (created in this session)
    2. Handoff Document v2.2.md - For reference
    3. Specification v1.5.md - For reference
Files for Real Data Testing (if desired)
    1. TickHotLoopF32.jl - For real tick data processing
    2. YM 06-25.Last.txt - Sample market data (if testing with real data)

Your tasks
    1. NOTE: YOU ARE INSTRUCTED TO NOTIFY ME IF YOU ENCOUNTER ANY Dictionary-Based Code. ANY SUCH CODE WILL NEED TO BE REWRITTEN TO USE DIRECT STRUCTS ONLY. No Dict() is allowed for data structures!
    2. review: project handoff document: 
        ○ GA Optimization System - PLL Filter Bank Handoff Document v2.1.md
    3. Review: project specification document:
        ○ GA Optimization System for ComplexBiquad PLL Filter Bank - Specification v1.5.md
    4. Request code files that you need to test Chunk 4.
    5. Modify the uploaded load_all.jl script to work with test_chunk4.jl to eliminate Include/Import load sequence issue. 
    6. Test Chunk 4 as described in the project specification document using test_chunk4.jl
    The revised test file includes ALL original test categories:
        • ✅ RMS-based weight initialization
        • ✅ Scalar weight application
        • ✅ Weight mutation and crossover
        • ✅ Multi-horizon support
        • ✅ Performance benchmarks
        • ✅ Integration tests
    PLUS adds:
        • ✅ Phase extrapolation tests
        • ✅ Frequency calculation tests
        • ✅ Mathematical validation
        • ✅ Phase coherence tests
