-- Pure functions extracted from telemetry.lua for testing
local M = {}

function M.get_pixel(data, stride, x, y)
    local offset = y * stride + x * 4
    if offset + 3 > #data or offset < 0 then return 0, 0, 0 end
    return data:byte(offset + 3), data:byte(offset + 2), data:byte(offset + 1)
end

function M.sample_bar(data, stride, cfg)
    if not cfg.x then return 0 end
    local last_active = -1
    for col = 0, cfg.w - 1 do
        local hits, total = 0, 0
        for row = 0, cfg.h - 1 do
            local r, g, b = M.get_pixel(data, stride, cfg.x + col, cfg.y + row)
            if cfg.color_channel == "saturation" then
                local mx = math.max(r, g, b)
                local mn = math.min(r, g, b)
                if mx > 60 and (mx > 0 and (mx - mn) / mx or 0) > (cfg.threshold or 0.2) then hits = hits + 1 end
            elseif cfg.color_channel == "red" then
                if r > (cfg.threshold or 140) then hits = hits + 1 end
            elseif cfg.color_channel == "green" then
                if g > (cfg.threshold or 80) then hits = hits + 1 end
            elseif cfg.color_channel == "blue" then
                if b > (cfg.threshold or 80) then hits = hits + 1 end
            end
            total = total + 1
        end
        if total > 0 and (hits / total) > 0.3 then last_active = col end
    end
    if last_active < 0 then return 0 end
    return math.min(1.0, (last_active + 1) / cfg.w)
end

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
    local patterns = {
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
    local best, best_s = 0, -1
    for d = 0, 9 do
        local s = 0
        for i = 0, 34 do if grid[i] == patterns[d][i + 1] then s = s + 1 end end
        if s > best_s then best_s = s; best = d end
    end
    return best
end

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
