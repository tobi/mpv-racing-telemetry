#!/usr/bin/env lua
-- Integration tests using real BGRA frames from TDS Racing IMSA Sebring video

package.path = package.path .. ";../?.lua"
local core = require("telemetry_core")

local pass, fail, total = 0, 0, 0
local function test(name, fn)
    total = total + 1
    local ok, err = pcall(fn)
    if ok then pass = pass + 1; io.write(string.format("  ✓ %s\n", name))
    else fail = fail + 1; io.write(string.format("  ✗ %s — %s\n", name, err)) end
end
local function near(a, b, eps, msg)
    eps = eps or 0.10
    if math.abs(a - b) > eps then
        error(string.format("%s: expected ~%.2f, got %.2f", msg or "near", b, a), 2)
    end
end

local STRIDE = 1280 * 4
local function load_frame(path)
    local f = io.open(path, "rb"); local d = f:read("*a"); f:close()
    assert(#d == 1280*720*4, path .. ": bad size " .. #d)
    return d
end

-- ── Bar configs ──
-- Throttle: bar spans x=1130..1213, y=620 is a clean row
-- filled pixels: g≈155-162 → active_color = (0,158,0)
-- unfilled pixels: g≈55-65 → dist from active ≈ 100
-- Use color_dist=55 so filled (dist~0) matches, unfilled (dist~100) doesn't
-- Filled=rgb(17,158,17) dist=24 from active | Unfilled=dist≥97 | White text=dist≥373
local THROTTLE = { type="bar", x=1130, y=612, w=84, h=16,
    active_r=0, active_g=158, active_b=0, color_dist=60 }

-- Filled=rgb(228,20,22) dist=10 from active | Unfilled bg=dist≥107 | White text=dist≥314
local BRAKE = { type="bar", x=1130, y=634, w=84, h=20,
    active_r=230, active_g=30, active_b=20, color_dist=60 }

-- Gear: white circle with dark digit carved out — inverted OCR
local GEAR = { type="digit", x=1083, y=637, w=8, h=12, invert=true, threshold=225 }

-- ══════════════════════════════════════════════════════════════
-- frame_0540: ~68% throttle, 0% brake, gear 2
-- ══════════════════════════════════════════════════════════════
print("frame_0540 (t=5:40 — ~68% throttle, 0% brake)")

local f0540 = load_frame("frame_0540.raw")

test("throttle ~68%", function()
    local val = core.sample_bar(f0540, STRIDE, THROTTLE)
    near(val, 0.68, 0.10, "throttle")
end)

test("brake 0%", function()
    local val = core.sample_bar(f0540, STRIDE, BRAKE)
    near(val, 0.0, 0.05, "brake")
end)

test("gear = 2", function()
    local val = core.sample_digit(f0540, STRIDE, GEAR.x, GEAR.y, GEAR.w, GEAR.h, GEAR)
    assert(val == 2, string.format("expected 2, got %d", val))
end)

-- ══════════════════════════════════════════════════════════════
-- frame_0558: 0% throttle, ~25% brake (bright fill), gear 3
-- (visually looks like more, but bright red only fills ~25% from left)
-- ══════════════════════════════════════════════════════════════
print("\nframe_0558 (t=6:38 — 0% throttle, braking)")

local f0558 = load_frame("frame_0558.raw")

test("throttle ~0%", function()
    local val = core.sample_bar(f0558, STRIDE, THROTTLE)
    near(val, 0.0, 0.05, "throttle")
end)

test("brake > 0% (active)", function()
    local val = core.sample_bar(f0558, STRIDE, BRAKE)
    assert(val > 0.10, string.format("expected brake > 10%%, got %.0f%%", val * 100))
end)

test("gear = 3", function()
    local val = core.sample_digit(f0558, STRIDE, GEAR.x, GEAR.y, GEAR.w, GEAR.h, GEAR)
    assert(val == 3, string.format("expected 3, got %d", val))
end)

-- ══════════════════════════════════════════════════════════════
-- frame_0559: 0% throttle, light braking, gear 2
-- ══════════════════════════════════════════════════════════════
print("\nframe_0559 (t=6:39 — 0% throttle, trail braking)")

local f0559 = load_frame("frame_0559.raw")

test("throttle ~0%", function()
    local val = core.sample_bar(f0559, STRIDE, THROTTLE)
    near(val, 0.0, 0.05, "throttle")
end)

test("brake > 0% (some braking)", function()
    local val = core.sample_bar(f0559, STRIDE, BRAKE)
    assert(val > 0.05, string.format("expected brake > 5%%, got %.0f%%", val * 100))
end)

test("gear = 2", function()
    local val = core.sample_digit(f0559, STRIDE, GEAR.x, GEAR.y, GEAR.w, GEAR.h, GEAR)
    assert(val == 2, string.format("expected 2, got %d", val))
end)

-- ══════════════════════════════════════════════════════════════
-- Gear detection across all captured gears
-- ══════════════════════════════════════════════════════════════
print("\ngear detection (all gears)")

local gear_frames = {
    { "frame_0370.raw", 1, "t=370 gear 1" },
    { "frame_0540.raw", 2, "t=340 gear 2" },
    { "frame_0558.raw", 3, "t=398 gear 3" },
    { "frame_0865.raw", 4, "t=865 gear 4" },
    { "frame_0550.raw", 5, "t=550 gear 5" },
    { "frame_0553.raw", 6, "t=553 gear 6" },
}

for _, info in ipairs(gear_frames) do
    test(info[3], function()
        local data = load_frame(info[1])
        local val = core.sample_digit(data, STRIDE, GEAR.x, GEAR.y, GEAR.w, GEAR.h, GEAR)
        assert(val == info[2], string.format("expected %d, got %d", info[2], val))
    end)
end

-- ══════════════════════════════════════════════════════════════
-- Cross-frame consistency: throttle in 0558/0559 should be less than in 0540
-- ══════════════════════════════════════════════════════════════
print("\ncross-frame consistency")

test("0540 throttle > 0558 throttle", function()
    local t0 = core.sample_bar(f0540, STRIDE, THROTTLE)
    local t1 = core.sample_bar(f0558, STRIDE, THROTTLE)
    assert(t0 > t1 + 0.3, string.format("0540=%.2f should be >> 0558=%.2f", t0, t1))
end)

test("0558 brake > 0540 brake", function()
    local b0 = core.sample_bar(f0540, STRIDE, BRAKE)
    local b1 = core.sample_bar(f0558, STRIDE, BRAKE)
    assert(b1 > b0 + 0.05, string.format("0558=%.2f should be > 0540=%.2f", b1, b0))
end)

test("0558 brake >= 0559 brake", function()
    local b1 = core.sample_bar(f0558, STRIDE, BRAKE)
    local b2 = core.sample_bar(f0559, STRIDE, BRAKE)
    assert(b1 >= b2 - 0.05, string.format("0558=%.2f should be >= 0559=%.2f", b1, b2))
end)

-- ══════════════════════════════════════════════════════════════
-- Debug: print actual values
-- ══════════════════════════════════════════════════════════════
print("\nactual values:")
for _, info in ipairs({
    {"0540", f0540}, {"0558", f0558}, {"0559", f0559}
}) do
    local t = core.sample_bar(info[2], STRIDE, THROTTLE)
    local b = core.sample_bar(info[2], STRIDE, BRAKE)
    print(string.format("  %s: throttle=%.0f%% brake=%.0f%%", info[1], t*100, b*100))
end

print(string.format("\n%d/%d passed, %d failed", pass, total, fail))
os.exit(fail > 0 and 1 or 0)
