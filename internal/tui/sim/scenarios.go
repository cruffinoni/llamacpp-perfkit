package sim

import "github.com/cruffinoni/llamacpp-perfkit/internal/domain"

// MixedScenarioreturns the default benchmark simulation scenario with 3
// servers, 13 prompts total, covering success, timeout, OOM, failed, running,
// and pending states.
func MixedScenario() Scenario {
	return Scenario{
		ModelName: "Qwen3.6-35B-A3B-MTP UD-Q4_K_M",
		Configs: []Server{
			server1(),
			server2(),
			server3(),
		},
	}
}

func server1() Server {
	return Server{
		ID:          "sim-server-001",
		ContextSize: 8192,
		KVType:      "q8_0",
		NCPUMOE:     18,
		SpecType:    "none",
		BatchSize:   512,
		UBatchSize:  512,
		Prompts: []Prompt{
			codePythonSuccess(),
			codeCppSuccess(),
			longCodeReviewSuccess(),
			longPrefill8kTimeout(),
			chatTranslateSuccess(),
		},
	}
}

func server2() Server {
	return Server{
		ID:          "sim-server-002",
		ContextSize: 16384,
		KVType:      "q4_0",
		NCPUMOE:     18,
		SpecType:    "draft-mtp",
		BatchSize:   512,
		UBatchSize:  512,
		Prompts: []Prompt{
			longPrefill32kSuccess(),
			longPrefill48kOOM(),
			codePythonSuccess(),
			summarizeDocFailed(),
			longPrefill60kSuccess(),
		},
	}
}

func server3() Server {
	return Server{
		ID:          "sim-server-003",
		ContextSize: 8192,
		KVType:      "q8_0",
		NCPUMOE:     0,
		SpecType:    "none",
		BatchSize:   1024,
		UBatchSize:  512,
		Prompts: []Prompt{
			codeCppSuccess(),
			longCodeReviewSuccess(),
			codePythonSuccess(),
		},
	}
}

// --- success prompts ---

func codePythonSuccess() Prompt {
	gGen, gDone := new(78.0), new(78.0)
	pPrefill, pGen, pDone := new(812.0), new(812.0), new(812.0)
	m1, m2, m3, m4 := new(5520.0), new(5520.0), new(5520.0), new(5520.0)
	return Prompt{
		Profile: "code_python",
		Steps: []Step{
			{Phase: domain. PhaseStarting, DurationSec: 0.10, MinVRAMMiB: m1, TickCount: 2},
			{Phase: domain. PhasePrefill, DurationSec: 0.75, PromptTokS: pPrefill, MinVRAMMiB: m2, TickCount: 4},
			{Phase: domain. PhaseGenerating, DurationSec: 2.10, PromptTokS: pGen, GenTokS: gGen, MinVRAMMiB: m3, TickCount: 8},
			{Phase: domain. PhaseDone, DurationSec: 2.70, PromptTokS: pDone, GenTokS: gDone, MinVRAMMiB: m4, TickCount: 1},
		},
	}
}

func codeCppSuccess() Prompt {
	gGen, gDone := new(63.2), new(63.2)
	pPrefill, pGen, pDone := new(691.0), new(691.0), new(691.0)
	m1, m2, m3, m4 := new(5417.0), new(5417.0), new(5417.0), new(5417.0)
	return Prompt{
		Profile: "code_cpp",
		Steps: []Step{
			{Phase: domain. PhaseStarting, DurationSec: 0.12, MinVRAMMiB: m1, TickCount: 2},
			{Phase: domain. PhasePrefill, DurationSec: 0.90, PromptTokS: pPrefill, MinVRAMMiB: m2, TickCount: 5},
			{Phase: domain. PhaseGenerating, DurationSec: 2.50, PromptTokS: pGen, GenTokS: gGen, MinVRAMMiB: m3, TickCount: 10},
			{Phase: domain. PhaseDone, DurationSec: 3.21, PromptTokS: pDone, GenTokS: gDone, MinVRAMMiB: m4, TickCount: 1},
		},
	}
}

func longCodeReviewSuccess() Prompt {
	gGen, gDone := new(68.2), new(68.2)
	pPrefill, pGen, pDone := new(604.0), new(604.0), new(604.0)
	m1, m2, m3, m4 := new(5448.0), new(5448.0), new(5448.0), new(5448.0)
	return Prompt{
		Profile: "long_code_review",
		Steps: []Step{
			{Phase: domain. PhaseStarting, DurationSec: 0.15, MinVRAMMiB: m1, TickCount: 3},
			{Phase: domain. PhasePrefill, DurationSec: 1.10, PromptTokS: pPrefill, MinVRAMMiB: m2, TickCount: 6},
			{Phase: domain. PhaseGenerating, DurationSec: 2.60, PromptTokS: pGen, GenTokS: gGen, MinVRAMMiB: m3, TickCount: 12},
			{Phase: domain. PhaseDone, DurationSec: 3.25, PromptTokS: pDone, GenTokS: gDone, MinVRAMMiB: m4, TickCount: 1},
		},
	}
}

func longPrefill32kSuccess() Prompt {
	gGen, gDone := new(79.3), new(79.3)
	pPrefill, pGen, pDone := new(902.0), new(902.0), new(902.0)
	m1, m2, m3, m4 := new(5509.0), new(5509.0), new(5509.0), new(5509.0)
	return Prompt{
		Profile: "long_prefill_32k",
		Steps: []Step{
			{Phase: domain. PhaseStarting, DurationSec: 0.18, MinVRAMMiB: m1, TickCount: 2},
			{Phase: domain. PhasePrefill, DurationSec: 1.50, PromptTokS: pPrefill, MinVRAMMiB: m2, TickCount: 6},
			{Phase: domain. PhaseGenerating, DurationSec: 3.10, PromptTokS: pGen, GenTokS: gGen, MinVRAMMiB: m3, TickCount: 10},
			{Phase: domain. PhaseDone, DurationSec: 3.85, PromptTokS: pDone, GenTokS: gDone, MinVRAMMiB: m4, TickCount: 1},
		},
	}
}

func longPrefill60kSuccess() Prompt {
	gGen, gDone := new(61.7), new(61.7)
	pPrefill, pGen, pDone := new(880.0), new(880.0), new(880.0)
	m1, m2, m3, m4 := new(5300.0), new(5300.0), new(5300.0), new(5300.0)
	return Prompt{
		Profile: "long_prefill_60k",
		Steps: []Step{
			{Phase: domain. PhaseStarting, DurationSec: 0.25, MinVRAMMiB: m1, TickCount: 2},
			{Phase: domain. PhasePrefill, DurationSec: 2.20, PromptTokS: pPrefill, MinVRAMMiB: m2, TickCount: 6},
			{Phase: domain. PhaseGenerating, DurationSec: 4.30, PromptTokS: pGen, GenTokS: gGen, MinVRAMMiB: m3, TickCount: 10},
			{Phase: domain. PhaseDone, DurationSec: 5.10, PromptTokS: pDone, GenTokS: gDone, MinVRAMMiB: m4, TickCount: 1},
		},
	}
}

func chatTranslateSuccess() Prompt {
	gGen, gDone := new(71.5), new(71.5)
	pPrefill, pGen, pDone := new(740.0), new(740.0), new(740.0)
	m1, m2, m3, m4 := new(5480.0), new(5480.0), new(5480.0), new(5480.0)
	return Prompt{
		Profile: "chat_translate",
		Steps: []Step{
			{Phase: domain. PhaseStarting, DurationSec: 0.11, MinVRAMMiB: m1, TickCount: 2},
			{Phase: domain. PhasePrefill, DurationSec: 0.60, PromptTokS: pPrefill, MinVRAMMiB: m2, TickCount: 4},
			{Phase: domain. PhaseGenerating, DurationSec: 1.80, PromptTokS: pGen, GenTokS: gGen, MinVRAMMiB: m3, TickCount: 6},
			{Phase: domain. PhaseDone, DurationSec: 2.35, PromptTokS: pDone, GenTokS: gDone, MinVRAMMiB: m4, TickCount: 1},
		},
	}
}

// --- failure prompts ---

func longPrefill8kTimeout() Prompt {
	gGen := new(42.0)
	pPrefill, pGen := new(755.0), new(755.0)
	m1, m2, m3 := new(5489.0), new(5489.0), new(5489.0)
	return Prompt{
		Profile: "long_prefill_8k",
		Steps: []Step{
			{Phase: domain. PhaseStarting, DurationSec: 0.14, MinVRAMMiB: m1, TickCount: 2},
			{Phase: domain. PhasePrefill, DurationSec: 1.80, PromptTokS: pPrefill, MinVRAMMiB: m2, TickCount: 8},
			{Phase: domain. PhaseGenerating, DurationSec: 8.50, PromptTokS: pGen, GenTokS: gGen, MinVRAMMiB: m3, TickCount: 4},
			{Phase: domain. PhaseTimeout, DurationSec: 30.00, TickCount: 1},
		},
	}
}

func longPrefill48kOOM() Prompt {
	gGen := new(35.1)
	pPrefill, pGen := new(820.0), new(820.0)
	m1, m2, m3 := new(5420.0), new(5420.0), new(5420.0)
	return Prompt{
		Profile: "long_prefill_48k",
		Steps: []Step{
			{Phase: domain. PhaseStarting, DurationSec: 0.22, MinVRAMMiB: m1, TickCount: 2},
			{Phase: domain. PhasePrefill, DurationSec: 2.00, PromptTokS: pPrefill, MinVRAMMiB: m2, TickCount: 4},
			{Phase: domain. PhaseGenerating, DurationSec: 4.20, PromptTokS: pGen, GenTokS: gGen, MinVRAMMiB: m3, TickCount: 3},
			{Phase: domain. PhaseOOM, DurationSec: 5.10, TickCount: 1},
		},
	}
}

func summarizeDocFailed() Prompt {
	m1 := new(5350.0)
	return Prompt{
		Profile: "summarize_doc",
		Steps: []Step{
			{Phase: domain. PhaseStarting, DurationSec: 0.08, MinVRAMMiB: m1, TickCount: 2},
			{Phase: domain. PhaseFailed, DurationSec: 0.42, TickCount: 1},
		},
	}
}
