-- telemetry_core.lua — pure functions, no mpv dependency
-- Used by telemetry.lua and testable standalone.

local M = {}

-- ── Pixel access ──

function M.get_pixel(data, stride, x, y)
    if x < 0 or y < 0 then return 0, 0, 0 end
    local offset = y * stride + x * 4
    if offset + 3 > #data or offset < 0 then return 0, 0, 0 end
    return data:byte(offset + 3), data:byte(offset + 2), data:byte(offset + 1) -- r, g, b
end

-- ── Bar sampling ──

-- Color distance (squared euclidean, no sqrt for speed)
function M.color_dist_sq(r1, g1, b1, r2, g2, b2)
    local dr, dg, db = r1 - r2, g1 - g2, b1 - b2
    return dr * dr + dg * dg + db * db
end

function M.sample_bar(data, stride, cfg)
    if not cfg.x then return 0 end

    -- Active color from calibration (picked by user clicking on the filled part)
    local ar = cfg.active_r or 200
    local ag = cfg.active_g or 0
    local ab = cfg.active_b or 0
    -- Max color distance² to count as "active" (default 60² = 3600)
    local max_dist_sq = (cfg.color_dist or 60) * (cfg.color_dist or 60)

    local last_active = -1
    for col = 0, cfg.w - 1 do
        -- Collect color distances for this column, take median to reject outliers (text, noise)
        local dists = {}
        for row = 0, cfg.h - 1 do
            local r, g, b = M.get_pixel(data, stride, cfg.x + col, cfg.y + row)
            dists[#dists + 1] = M.color_dist_sq(r, g, b, ar, ag, ab)
        end
        table.sort(dists)
        local median = dists[math.floor(#dists / 2) + 1] or 999999

        if median < max_dist_sq then
            last_active = col
        end
    end

    if last_active < 0 then return 0 end
    return math.min(1.0, (last_active + 1) / cfg.w)
end

-- ── Center-offset sampling ──

function M.sample_center_offset(data, stride, cfg)
    if not cfg.x then return 0 end
    local hits = {}
    for i = 0, cfg.w - 1 do hits[i] = 0 end
    for col = 0, cfg.w - 1 do
        for row = 0, cfg.h - 1 do
            local r, g, b = M.get_pixel(data, stride, cfg.x + col, cfg.y + row)
            if (r + g + b) / 3 > 150 then hits[col] = hits[col] + 1 end
        end
    end
    local max_h = 0
    for i = 0, cfg.w - 1 do if hits[i] > max_h then max_h = hits[i] end end
    if max_h < 2 then return 0 end
    local wsum, wtot = 0, 0
    for i = 0, cfg.w - 1 do
        if hits[i] > max_h * 0.4 then
            wsum = wsum + (cfg.x + i) * hits[i]; wtot = wtot + hits[i]
        end
    end
    if wtot == 0 then return 0 end
    local cx = cfg.center_x or (cfg.x + cfg.w / 2)
    return math.max(-1, math.min(1, (wsum / wtot - cx) / (cfg.w / 2)))
end

-- ── Digit OCR ──

M.DIGIT_PATTERNS = {
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

function M.sample_digit(data, stride, x, y, w, h)
    if not x then return 0 end
    local gw, gh = 5, 7
    local cw, ch = w / gw, h / gh
    local grid = {}
    for gy = 0, gh - 1 do
        for gx = 0, gw - 1 do
            local s, c = 0, 0
            for dy = 0, math.max(0, math.floor(ch) - 1) do
                for dx = 0, math.max(0, math.floor(cw) - 1) do
                    local r, g, b = M.get_pixel(data, stride,
                        math.floor(x + gx * cw + dx), math.floor(y + gy * ch + dy))
                    s = s + (r + g + b) / 3; c = c + 1
                end
            end
            grid[gy * gw + gx] = (c > 0 and s / c > 128) and 1 or 0
        end
    end
    local best, best_s = 0, -1
    for d = 0, 9 do
        local s = 0
        for i = 0, 34 do if grid[i] == M.DIGIT_PATTERNS[d][i + 1] then s = s + 1 end end
        if s > best_s then best_s = s; best = d end
    end
    return best
end

-- ── ASS formatting ──

function M.ass_color(r, g, b)
    return string.format("\\1c&H%02X%02X%02X&", b, g, r)
end

function M.ass_bord_color(r, g, b)
    return string.format("\\3c&H%02X%02X%02X&", b, g, r)
end

function M.ass_alpha(a)
    return string.format("\\1a&H%02X&", math.floor((1 - a) * 255))
end

return M
