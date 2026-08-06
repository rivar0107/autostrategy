"""Tests for design document quality check."""

from autostrategy.core.quality_check import DesignQualityCheck


def test_valid_design_passes():
    design = """
# 双均线策略

## 策略概述

基于均线交叉。

## 买入条件

- 金叉

## 卖出条件

- 死叉

## 止损

- 5% 止损

## 仓位管理

- 满仓
"""
    report = DesignQualityCheck.check(design)
    assert report.passed
    assert not report.errors


def test_missing_sections_fails():
    design = "# 策略\n\n## 策略概述\n\n"
    report = DesignQualityCheck.check(design)
    assert not report.passed
    assert any("买入条件" in error for error in report.errors)
    assert any("卖出条件" in error for error in report.errors)
