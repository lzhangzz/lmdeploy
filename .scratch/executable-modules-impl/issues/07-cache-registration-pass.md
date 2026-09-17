# 07: Make the cache-registration pass real

**What to build:** The registration walk of the spec's lifecycle becomes real: modules register their per-layer cache byte requirements with the cache registry during the dedicated pass, and the engine enforces the ordering — registration completes before category finalization, bind before the first run. At this ticket's close the walk is wired and exercised by the registry's own accounting; the attention and GDN modules that actually register arrive in the next two tickets.

**Blocked by:** 03 (Linear executable module).

**Status:** ready-for-agent

- [ ] Registration walk runs after prepare and before category finalization, driven engine-side (not by Python loops)
- [ ] Modules that register ranges keep only their offsets; the registry maps categories to object ids after all registrations
- [ ] Models whose modules register nothing load and run unchanged
- [ ] Build green; core-model smoke unchanged
