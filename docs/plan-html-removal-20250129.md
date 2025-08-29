# HTML Functionality Removal Action Plan

## Overview
This document outlines the comprehensive plan to remove all HTML-related functionality from the splurge-lazyframe-compare library. The HTML features include export capabilities, report generation, and security features that are no longer needed.

## Current HTML Functionality Inventory

### Core Components with HTML Features

#### 1. **ComparisonOrchestrator** (`splurge_lazyframe_compare/services/orchestrator.py`)
- `_escape_html()` - Static method for HTML entity escaping (XSS prevention)
- `export_result_to_html()` - Public method to export results to HTML file
- `_export_result_to_html_file()` - Private method for HTML file export
- `_generate_html_report()` - Private method generating HTML content (450+ lines)
- HTML report type handling in `generate_report_from_result()`

#### 2. **LazyFrameComparator** (`splurge_lazyframe_compare/core/comparator.py`)
- `export_to_html()` - Public method delegating to orchestrator

#### 3. **Documentation and Examples**
- `README.md` - Multiple HTML export references and examples
- `examples/README.md` - HTML functionality mentions
- `examples/tabulated_report_example.py` - HTML export demonstrations
- `plans/polars_comparison_framework_plan.md` - HTML export planning

#### 4. **Test Files**
- `tests/test_comprehensive_edge_cases.py` - `TestHTMLSecurity` class with 4 test methods
- `tests/test_services.py` - HTML export test methods
- `tests/test_comparator.py` - HTML export test methods

## Action Plan

### Phase 1: Code Removal (High Priority)

#### 1.1 Remove HTML Methods from ComparisonOrchestrator
**Files to modify:**
- `splurge_lazyframe_compare/services/orchestrator.py`

**Methods to remove:**
- `_escape_html()` (lines 17-42)
- `export_result_to_html()` (lines 247-263)
- `_export_result_to_html_file()` (lines 265-281)
- `_generate_html_report()` (lines 283-495, ~200 lines of HTML template)

**Modifications needed:**
- Update `generate_report_from_result()` to remove HTML report type handling (lines 238-240)
- Update method docstring to remove HTML reference (line 217)

#### 1.2 Remove HTML Method from LazyFrameComparator
**Files to modify:**
- `splurge_lazyframe_compare/core/comparator.py`

**Methods to remove:**
- `export_to_html()` (lines 223-242)

#### 1.3 Remove HTML References from ReportingService
**Files to check:**
- `splurge_lazyframe_compare/services/reporting_service.py`

**Note:** Current implementation shows HTML reporting as "not yet implemented" (returns summary report), so only documentation updates may be needed.

### Phase 2: Test Removal (Medium Priority)

#### 2.1 Remove HTML Security Tests
**Files to modify:**
- `tests/test_comprehensive_edge_cases.py`

**Tests to remove:**
- `TestHTMLSecurity` class (entire class, ~50 lines)
  - `test_html_escaping_prevents_xss()`
  - `test_html_escaping_normal_text()`
  - `test_html_escaping_edge_cases()`

#### 2.2 Remove HTML Export Tests
**Files to modify:**
- `tests/test_services.py`
- `tests/test_comparator.py`

**Test methods to remove:**
- `test_export_result_to_html_comprehensive()`
- `test_export_to_html_comprehensive()`
- `test_export_to_html_no_result_error()`

### Phase 3: Documentation Updates (Medium Priority)

#### 3.1 Update README.md
**Sections to modify:**
- Feature list (line 19) - Remove "and HTML formats"
- HTML Export section (lines 244-266) - Remove entire section
- Examples section - Remove HTML export references
- Tabulated report example description (line 869) - Remove HTML reference

#### 3.2 Update Example Documentation
**Files to modify:**
- `examples/README.md`

**Updates needed:**
- Remove HTML export references from example descriptions

### Phase 4: Example Updates (Low Priority)

#### 4.1 Update Tabulated Report Example
**Files to modify:**
- `examples/tabulated_report_example.py`

**Updates needed:**
- Remove HTML export demonstrations
- Update example descriptions and comments

### Phase 5: Planning Documentation Cleanup

#### 5.1 Update Planning Documents
**Files to modify:**
- `plans/polars_comparison_framework_plan.md`

**Updates needed:**
- Remove HTML export references from planning documentation

### Phase 6: Verification and Testing

#### 6.1 Code Verification
- Search for any remaining HTML references in codebase
- Verify no broken imports or method calls
- Test that library still functions without HTML features

#### 6.2 Dependency Check
- Verify no HTML-specific dependencies need removal
- Check for any HTML-related imports that can be cleaned up

## Implementation Strategy

### Recommended Order of Execution

1. **Start with core code removal** (Phase 1) to prevent breaking changes during development
2. **Remove tests** (Phase 2) to avoid test failures during refactoring
3. **Update documentation** (Phase 3) to reflect the new feature set
4. **Clean up examples** (Phase 4) for consistency
5. **Final verification** (Phase 6) to ensure completeness

### Risk Mitigation

- **Backup strategy:** Create git branch before starting removal
- **Incremental commits:** Commit after each major component removal
- **Testing:** Run full test suite after each phase
- **Documentation:** Update internal documentation as changes are made

### Success Criteria

- ✅ No HTML-related code remains in the codebase
- ✅ All tests pass without HTML test cases
- ✅ Documentation accurately reflects current functionality
- ✅ Examples demonstrate only supported features
- ✅ No broken imports or method references

## Timeline Estimate

- **Phase 1 (Code Removal):** 2-3 hours
- **Phase 2 (Test Removal):** 1-2 hours
- **Phase 3 (Documentation):** 1-2 hours
- **Phase 4 (Examples):** 1 hour
- **Phase 5 (Planning Cleanup):** 30 minutes
- **Phase 6 (Verification):** 1-2 hours

**Total estimated time:** 6-10 hours

## Post-Removal Considerations

### Feature Communication
- Update any external documentation or wikis
- Notify users if this is a breaking change
- Consider version bump if significant functionality removal

### Future Considerations
- Document this removal in changelog/release notes
- Consider adding deprecation warnings before complete removal (if phased approach preferred)
- Evaluate if HTML functionality should be moved to separate package if needed by users

---

*Document created: 2025-01-29*
*Last updated: 2025-01-29*
