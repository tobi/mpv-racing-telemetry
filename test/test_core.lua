#!/usr/bin/env lua
-- Unit tests for telemetry_core.lua

package.path = package.path .. ";../?.lua"
local core = require("telemetry_core")

-- ── tiny test harness ──────────────────────────────────────────
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

local function eq(a, b, msg)
    if a ~= b then error(string.format("%s: expected %s, got %s", msg or "eq", tostring(b), tostring(a)), 2) end
end

local function near(a, b, eps, msg)
    eps = eps or 0.01
    if math.abs(a - b) > eps then error(string.format("%s: expected ~%s, got %s", msg or "near", tostring(b), tostring(a)), 2) end
end

-- ── helpers to build fake pixel buffers ────────────────────────
-- BGRA format: byte order is B, G, R, A per pixel
local function make_image(width, height, pixel_fn)
    local bytes = {}
    for y = 0, height - 1 do
        for x = 0, width - 1 do
            local r, g, b = pixel_fn(x, y)
            bytes[#bytes + 1] = string.char(b, g, r, 255)
        end
    end
    return table.concat(bytes), width * 4
end

-- ── get_pixel ──────────────────────────────────────────────────
print("get_pixel")

test("reads BGRA pixel correctly", function()
    local data, stride = make_image(2, 2, function(x, y)
        if x == 1 and y == 0 then return 100, 150, 200 end
        return 0, 0, 0
    end)
    local r, g, b = core.get_pixel(data, stride, 1, 0)
    eq(r, 100, "r"); eq(g, 150, "g"); eq(b, 200, "b")
end)

test("out of bounds returns 0,0,0", function()
    local data, stride = make_image(2, 2, function() return 255, 255, 255 end)
    local r, g, b = core.get_pixel(data, stride, 10, 10)
    eq(r, 0, "r"); eq(g, 0, "g"); eq(b, 0, "b")
end)

test("negative coords return 0,0,0", function()
    local data, stride = make_image(2, 2, function() return 255, 255, 255 end)
    local r, g, b = core.get_pixel(data, stride, -1, 0)
    eq(r, 0, "r"); eq(g, 0, "g"); eq(b, 0, "b")
end)

-- ── sample_bar ─────────────────────────────────────────────────
print("\nsample_bar")

test("returns 0 when cfg.x is nil", function()
    eq(core.sample_bar("", 0, {w = 10, h = 1}), 0)
end)

test("full red bar returns 1.0", function()
    local data, stride = make_image(20, 2, function() return 255, 0, 0 end)
    local val = core.sample_bar(data, stride, {x = 0, y = 0, w = 20, h = 2, color_channel = "red", threshold = 140})
    eq(val, 1.0, "full bar")
end)

test("empty bar returns 0", function()
    local data, stride = make_image(20, 2, function() return 0, 0, 0 end)
    local val = core.sample_bar(data, stride, {x = 0, y = 0, w = 20, h = 2, color_channel = "red", threshold = 140})
    eq(val, 0, "empty bar")
end)

test("half red bar returns ~0.5", function()
    local data, stride = make_image(20, 2, function(x)
        if x < 10 then return 255, 0, 0 else return 0, 0, 0 end
    end)
    local val = core.sample_bar(data, stride, {x = 0, y = 0, w = 20, h = 2, color_channel = "red", threshold = 140})
    near(val, 0.5, 0.05, "half bar")
end)

test("blue channel detection", function()
    local data, stride = make_image(10, 1, function() return 0, 0, 200 end)
    local val = core.sample_bar(data, stride, {x = 0, y = 0, w = 10, h = 1, color_channel = "blue", threshold = 80})
    eq(val, 1.0, "full blue")
end)

test("green channel detection", function()
    local data, stride = make_image(10, 1, function() return 0, 200, 0 end)
    local val = core.sample_bar(data, stride, {x = 0, y = 0, w = 10, h = 1, color_channel = "green", threshold = 80})
    eq(val, 1.0, "full green")
end)

test("saturation detection", function()
    local data, stride = make_image(10, 1, function() return 255, 0, 0 end)
    local val = core.sample_bar(data, stride, {x = 0, y = 0, w = 10, h = 1, color_channel = "saturation", threshold = 0.2})
    eq(val, 1.0, "saturated")
end)

test("saturation rejects gray", function()
    local data, stride = make_image(10, 1, function() return 128, 128, 128 end)
    local val = core.sample_bar(data, stride, {x = 0, y = 0, w = 10, h = 1, color_channel = "saturation", threshold = 0.2})
    eq(val, 0, "gray not saturated")
end)

-- ── sample_center_offset ───────────────────────────────────────
print("\nsample_center_offset")

test("returns 0 when cfg.x is nil", function()
    eq(core.sample_center_offset("", 0, {w = 10, h = 1}), 0)
end)

test("centered bright stripe returns ~0", function()
    local w = 20
    local data, stride = make_image(w, 4, function(x)
        if x >= 8 and x <= 12 then return 255, 255, 255 else return 0, 0, 0 end
    end)
    local val = core.sample_center_offset(data, stride, {x = 0, y = 0, w = w, h = 4, center_x = 10})
    near(val, 0, 0.15, "centered")
end)

test("right-shifted stripe returns positive", function()
    local w = 20
    local data, stride = make_image(w, 4, function(x)
        if x >= 15 and x <= 19 then return 255, 255, 255 else return 0, 0, 0 end
    end)
    local val = core.sample_center_offset(data, stride, {x = 0, y = 0, w = w, h = 4, center_x = 10})
    assert(val > 0.3, string.format("expected positive, got %s", val))
end)

test("left-shifted stripe returns negative", function()
    local w = 20
    local data, stride = make_image(w, 4, function(x)
        if x >= 0 and x <= 4 then return 255, 255, 255 else return 0, 0, 0 end
    end)
    local val = core.sample_center_offset(data, stride, {x = 0, y = 0, w = w, h = 4, center_x = 10})
    assert(val < -0.3, string.format("expected negative, got %s", val))
end)

test("all dark returns 0", function()
    local data, stride = make_image(20, 4, function() return 0, 0, 0 end)
    local val = core.sample_center_offset(data, stride, {x = 0, y = 0, w = 20, h = 4, center_x = 10})
    eq(val, 0, "all dark")
end)

-- ── sample_digit ───────────────────────────────────────────────
print("\nsample_digit")

test("returns 0 when x is nil", function()
    eq(core.sample_digit("", 0, nil, 0, 10, 14), 0)
end)

local function digit_image(pattern, x, y, w, h)
    local gw, gh = 5, 7
    local cw, ch = w / gw, h / gh
    local iw, ih = x + w + 2, y + h + 2
    local data, stride = make_image(iw, ih, function(px, py)
        local lx, ly = px - x, py - y
        if lx < 0 or ly < 0 or lx >= w or ly >= h then return 0, 0, 0 end
        local gx = math.min(gw - 1, math.floor(lx / cw))
        local gy = math.min(gh - 1, math.floor(ly / ch))
        if pattern[gy * gw + gx + 1] == 1 then return 255, 255, 255 else return 0, 0, 0 end
    end)
    return data, stride
end

local DIGIT_PATTERNS = {
    [0] = {0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0},
    [1] = {0,0,1,0,0, 0,1,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0},
    [2] = {0,1,1,1,0, 1,0,0,0,1, 0,0,0,0,1, 0,0,1,1,0, 0,1,0,0,0, 1,0,0,0,0, 1,1,1,1,1},
    [3] = {0,1,1,1,0, 1,0,0,0,1, 0,0,0,0,1, 0,0,1,1,0, 0,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0},
    [4] = {0,0,0,1,0, 0,0,1,1,0, 0,1,0,1,0, 1,0,0,1,0, 1,1,1,1,1, 0,0,0,1,0, 0,0,0,1,0},
    [5] = {1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,0, 0,0,0,0,1, 0,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0},
    [6] = {0,1,1,1,0, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0},
    [7] = {1,1,1,1,1, 0,0,0,0,1, 0,0,0,1,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0},
    [8] = {0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0},
    [9] = {0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,1, 0,0,0,0,1, 0,0,0,0,1, 0,1,1,1,0},
}

for d = 0, 9 do
    test(string.format("recognizes digit %d", d), function()
        local x, y, w, h = 0, 0, 25, 35
        local data, stride = digit_image(DIGIT_PATTERNS[d], x, y, w, h)
        local result = core.sample_digit(data, stride, x, y, w, h)
        eq(result, d, string.format("digit %d", d))
    end)
end

-- ── ASS formatting ─────────────────────────────────────────────
print("\nass formatting")

test("ass_color formats BGR", function()
    eq(core.ass_color(255, 0, 0), "\\1c&H0000FF&")
    eq(core.ass_color(0, 255, 0), "\\1c&H00FF00&")
    eq(core.ass_color(0, 0, 255), "\\1c&HFF0000&")
end)

test("ass_bord_color formats BGR", function()
    eq(core.ass_bord_color(255, 128, 0), "\\3c&H0080FF&")
end)

test("ass_alpha fully opaque", function()
    eq(core.ass_alpha(1.0), "\\1a&H00&")
end)

test("ass_alpha fully transparent", function()
    eq(core.ass_alpha(0.0), "\\1a&HFF&")
end)

test("ass_alpha half transparent", function()
    eq(core.ass_alpha(0.5), "\\1a&H7F&")
end)

-- ── summary ────────────────────────────────────────────────────
print(string.format("\n%d/%d passed, %d failed", pass, total, fail))
os.exit(fail > 0 and 1 or 0)
