#!/usr/bin/env bash
celery -A AnalysisEngine worker -B -l info
