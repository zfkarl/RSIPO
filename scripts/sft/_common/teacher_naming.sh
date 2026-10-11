#!/usr/bin/env bash

infer_sft_teacher_short() {
    local explicit="${SFT_TEACHER_SHORT:-${TEACHER_MODEL_SHORT:-${TEACHER_SHORT:-}}}"
    if [[ -n "$explicit" ]]; then
        printf '%s' "$explicit" | tr '[:upper:]' '[:lower:]'
        return
    fi

    local model="${SFT_TEACHER_MODEL:-${TEACHER_MODEL:-${SKILL_MODEL:-${OPENAI_MODEL:-}}}}"
    local lowered
    lowered="$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]')"
    case "$lowered" in
        *gpt-5.6*|*gpt5.6*|*gpt_5.6*|*gpt56*)
            printf 'gpt56'
            ;;
        *deepseek*|*deep-seek*|*ds*)
            printf 'ds'
            ;;
        *gemini*)
            printf 'gemini'
            ;;
        *qwen*)
            printf 'qwen'
            ;;
        *glm*)
            printf 'glm'
            ;;
        *)
            printf 'teacher'
            ;;
    esac
}

SFT_TEACHER_SHORT="$(infer_sft_teacher_short)"
SFT_TEACHER_DIR_SHORT="${SFT_TEACHER_SHORT//-/_}"
SFT_SELF_SUFFIX="${SFT_TEACHER_SHORT}-self"
SFT_SELF_DIR_SUFFIX="${SFT_TEACHER_DIR_SHORT}_self"
