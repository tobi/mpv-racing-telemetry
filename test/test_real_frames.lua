#!/usr/bin/env lua
-- Integration tests using real video frame data (BGRA raw frames)

package.path = package.path .. ";../?.lua"
local core = require("telemetry_core")

local pass, fail, total = 0, 0, 0
local function test(name, fn)
    total = total + 1
    local ok, err = pcall(fn)
    if ok then
        pass = pass + 1
        io.write(string.format("  ✓ %s\n", name))
    else
        fail = fail + 1
        io.write(string.format("  ✗ %s — %s\n", name, err))
    end
end

local function near(a, b, eps, msg)
    eps = eps or 0.05
    if math.abs(a - b) > eps then
        error(string.format("%s: expected ~%.2f, got %.2f (delta %.2f)", msg or "near", b, a, math.abs(a-b)), 2)
    end
end

local STRIDE = 1280 * 4

local function load_frame(path)
    local f = io.open(path, "rb")
    if not f then error("Cannot open " .. path) end
    local data = f:read("*a")
    f:close()
    assert(#data == 1280 * 720 * 4, "Expected 1280x720 BGRA frame, got " .. #data .. " bytes")
    return data
end

-- Measurement configs matching the TDS Racing IMSA overlay
local THROTTLE = { type = "bar", x = 1130, y = 606, w = 82, h = 18,
    active_r = 0, active_g = 80, active_b = 0, color_dist = 35 }

local BRAKE = { type = "bar", x = 1130, y = 634, w = 82, h = 20,
    active_r = 220, active_g = 30, active_b = 20, color_dist = 60 }

local GEAR = { type = "digit", x = 1088, y = 625, w = 30, h = 30 }

-- ── frame_0540: ~70% throttle, 0% brake, gear 2 ──
print("frame_0540 (t=5:40 — 70% throttle, 0% brake, gear 2)")

local frame_0540 = load_frame("frame_0540.raw")

test("throttle ~70%", function()
    local val = core.sample_bar(frame_0540, STRIDE, THROTTLE)
    near(val, 0.70, 0.12, "throttle")
end)

test("brake 0%", function()
    local val = core.sample_bar(frame_0540, STRIDE, BRAKE)
    near(val, 0.0, 0.05, "brake")
end)

test("gear = 2", function()
    local val = core.sample_digit(frame_0540, STRIDE, GEAR.x, GEAR.y, GEAR.w, GEAR.h)
    assert(val == 2, string.format("expected gear 2, got %d", val))
end)

-- ── frame_0558: ~0% throttle, ~80% brake, gear 3 ──
print("\nframe_0558 (t=6:38 — 0% throttle, ~80% brake, gear 3)")

local frame_0558 = load_frame("frame_0558.raw")

test("throttle ~0%", function()
    local val = core.sample_bar(frame_0558, STRIDE, THROTTLE)
    near(val, 0.0, 0.08, "throttle")
end)

test("brake ~80%", function()
    local val = core.sample_bar(frame_0558, STRIDE, BRAKE)
    near(val, 0.80, 0.15, "brake")
end)

test("gear = 3", function()
    local val = core.sample_digit(frame_0558, STRIDE, GEAR.x, GEAR.y, GEAR.w, GEAR.h)
    assert(val == 3, string.format("expected gear 3, got %d", val))
end)

-- ── frame_0559: ~0% throttle, ~25% brake, gear 2 ──
print("\nframe_0559 (t=6:39 — trail braking, ~25% brake, gear 2)")

local frame_0559 = load_frame("frame_0559.raw")

test("throttle ~0%", function()
    local val = core.sample_bar(frame_0559, STRIDE, THROTTLE)
    near(val, 0.0, 0.08, "throttle")
end)

test("brake ~25%", function()
    local val = core.sample_bar(frame_0559, STRIDE, BRAKE)
    near(val, 0.25, 0.15, "brake")
end)

test("gear = 2", function()
    local val = core.sample_digit(frame_0559, STRIDE, GEAR.x, GEAR.y, GEAR.w, GEAR.h)
    assert(val == 2, string.format("expected gear 2, got %d", val))
end)

-- ── Summary ──
print(string.format("\n%d/%d passed, %d failed", pass, total, fail))
os.exit(fail > 0 and 1 or 0)
