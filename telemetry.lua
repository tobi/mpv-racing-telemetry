-- Racing Telemetry — mpv Lua script
--
-- Reads baked-in telemetry from onboard racing videos by sampling
-- raw pixel data in memory (screenshot-raw). Renders via ASS overlay.
--
-- Keys:
--   Ctrl+t       — toggle telemetry overlay
--   Ctrl+c       — enter/exit calibration mode
--   Ctrl+g       — cycle overlay position
--   Ctrl+= / -   — resize overlay
--   Ctrl+n       — cycle through saved calibrations
--
-- Calibration mode:
--   1-6          — select channel (1=throttle 2=brake 3=gear 4=steering 5=speed 6=fuel)
--   Click+drag   — draw rectangle for selected channel
--   c            — pick color (click on active/filled color)
--   m            — set center point (for steering)
--   s            — save calibration (prompts for name)
--   Escape/Space — exit calibration
--
-- Configs saved to ~/.config/mpv/telemetry-configs/

local msg = require("mp.msg")
local assdraw = require("mp.assdraw")
local utils = require("mp.utils")

-- Add script directory to Lua path so we can require telemetry_core
local script_dir = debug.getinfo(1, "S").source:match("@?(.*/)") or "./"
package.path = script_dir .. "?.lua;" .. package.path
local core = require("telemetry_core")

-- ══════════════════════════════════════════════════════════════
-- CONFIG PERSISTENCE
-- ══════════════════════════════════════════════════════════════

local CONFIG_DIR = (os.getenv("HOME") or "/tmp") .. "/.config/mpv/telemetry-configs"
local CHANNELS = {"throttle", "brake", "gear", "steering", "speed", "fuel"}

local CHANNEL_DEFAULTS = {
    throttle = { type = "bar", active_r = 0, active_g = 200, active_b = 0, color_dist = 80 },
    brake    = { type = "bar", active_r = 200, active_g = 0, active_b = 0, color_dist = 60 },
    gear     = { type = "digit" },
    steering = { type = "center-offset" },
    speed    = { type = "digits" },
    fuel     = { type = "bar", active_r = 0, active_g = 100, active_b = 255, color_dist = 60 },
}

-- Default config for TDS Racing IMSA (1280x720)
local DEFAULT_CONFIG = {
    throttle = { type = "bar", x = 1130, y = 612, w = 84, h = 16, active_r = 0, active_g = 158, active_b = 0, color_dist = 60 },
    brake    = { type = "bar", x = 1130, y = 634, w = 84, h = 20, active_r = 230, active_g = 30, active_b = 20, color_dist = 60 },
    gear     = { type = "digit", x = 1088, y = 625, w = 30, h = 30 },
    steering = { type = "center-offset", x = 110, y = 690, w = 110, h = 12, center_x = 165 },
}

-- Active config
local config = {}
for k, v in pairs(DEFAULT_CONFIG) do
    config[k] = {}
    for k2, v2 in pairs(v) do config[k][k2] = v2 end
end
local config_name = "default"  -- name of loaded config
local all_config_names = {} -- list of saved config names
local config_idx = 0        -- index into all_config_names (0 = none)

local function ensure_config_dir()
    os.execute('mkdir -p "' .. CONFIG_DIR .. '"')
end

local function list_configs()
    ensure_config_dir()
    local names = {}
    local p = io.popen('ls "' .. CONFIG_DIR .. '"/*.json 2>/dev/null')
    if p then
        for line in p:lines() do
            local name = line:match("([^/]+)%.json$")
            if name then names[#names + 1] = name end
        end
        p:close()
    end
    table.sort(names)
    return names
end

local function save_config(name)
    ensure_config_dir()
    local path = CONFIG_DIR .. "/" .. name .. ".json"
    local entries = {}
    for ch, cfg in pairs(config) do
        -- Serialize each channel config
        local e = { channel = ch }
        for k, v in pairs(cfg) do e[k] = v end
        entries[#entries + 1] = e
    end
    -- Simple JSON serializer (no external deps)
    local function to_json(val, indent)
        indent = indent or 0
        local pad = string.rep("  ", indent)
        local pad1 = string.rep("  ", indent + 1)
        if type(val) == "table" then
            if #val > 0 or next(val) == nil then
                -- array
                if #val == 0 then return "[]" end
                local items = {}
                for _, v in ipairs(val) do items[#items + 1] = pad1 .. to_json(v, indent + 1) end
                return "[\n" .. table.concat(items, ",\n") .. "\n" .. pad .. "]"
            else
                -- object
                local items = {}
                local keys = {}
                for k in pairs(val) do keys[#keys + 1] = k end
                table.sort(keys)
                for _, k in ipairs(keys) do
                    items[#items + 1] = pad1 .. '"' .. k .. '": ' .. to_json(val[k], indent + 1)
                end
                return "{\n" .. table.concat(items, ",\n") .. "\n" .. pad .. "}"
            end
        elseif type(val) == "string" then
            return '"' .. val:gsub('"', '\\"') .. '"'
        elseif type(val) == "number" then
            return tostring(val)
        elseif type(val) == "boolean" then
            return val and "true" or "false"
        else
            return '"' .. tostring(val) .. '"'
        end
    end
    local json = to_json({ name = name, measurements = entries })
    local f = io.open(path, "w")
    if f then
        f:write(json .. "\n")
        f:close()
        msg.info("Saved config: " .. path)
        config_name = name
        all_config_names = list_configs()
        for i, n in ipairs(all_config_names) do
            if n == name then config_idx = i; break end
        end
    else
        msg.error("Could not write: " .. path)
    end
end

local function load_config(name)
    local path = CONFIG_DIR .. "/" .. name .. ".json"
    local f = io.open(path, "r")
    if not f then return false end
    local raw = f:read("*a")
    f:close()

    -- Parse JSON (use mpv's built-in)
    local parsed = utils.parse_json(raw)
    if not parsed or not parsed.measurements then
        msg.warn("Bad config: " .. path)
        return false
    end

    config = {}
    for _, m in ipairs(parsed.measurements) do
        local ch = m.channel
        if ch then
            local cfg = {}
            for k, v in pairs(m) do
                if k ~= "channel" then cfg[k] = v end
            end
            config[ch] = cfg
        end
    end
    config_name = name
    msg.info("Loaded config: " .. name .. " (" .. #parsed.measurements .. " channels)")
    return true
end

local function load_most_recent()
    all_config_names = list_configs()
    if #all_config_names > 0 then
        local name = all_config_names[#all_config_names]
        if load_config(name) then
            config_idx = #all_config_names
            return true
        end
    end
    return false
end

-- ══════════════════════════════════════════════════════════════
-- STATE
-- ══════════════════════════════════════════════════════════════

local overlay_visible = true
local overlay = mp.create_osd_overlay("ass-events")
local cal_overlay = mp.create_osd_overlay("ass-events") -- calibration overlay
local timer = nil

-- Overlay layout
local widget_pos = { x = 0.5, y = 0.92 }
local scale = 1.0
local last_widget_rect = { x = 0, y = 0, w = 0, h = 0 }

-- Drag state
local dragging = false
local drag_offset = { x = 0, y = 0 }

-- Trace history
local TRACE_LEN = 200
local trace = {}
local trace_idx = 0

-- Current values
local cur = { throttle = 0, brake = 0, gear = 0, steering = 0, speed = 0, fuel = 0 }
local raw_vals = { throttle = 0, brake = 0, gear = 0, steering = 0, speed = 0, fuel = 0 }
local SMOOTH = 0.3

-- Forward declarations
local enter_calibration, exit_calibration, render_calibration, render_overlay

-- Calibration mode
local cal_active = false
local cal_channel = nil       -- selected channel name
local cal_channel_idx = 0     -- 1-6
local cal_mode = "draw"       -- "draw", "pick-color", "pick-center"
local cal_drawing = false
local cal_draw_start = nil    -- {x, y} in video pixels
local cal_draw_cur = nil
local cal_last_screenshot = nil -- {data, stride, w, h}

-- ══════════════════════════════════════════════════════════════
-- PIXEL SAMPLING (same as before)
-- ══════════════════════════════════════════════════════════════

local get_pixel = core.get_pixel

local sample_bar = core.sample_bar

local sample_center_offset = core.sample_center_offset

local sample_digit = core.sample_digit

-- ══════════════════════════════════════════════════════════════
-- FRAME PROCESSING
-- ══════════════════════════════════════════════════════════════

local function process_frame()
    local res = mp.command_native({"screenshot-raw", "video"})
    if not res or not res.data then return end
    local data, stride = res.data, res.stride

    -- Cache for calibration
    cal_last_screenshot = { data = data, stride = stride, w = res.w, h = res.h }

    for _, ch in ipairs(CHANNELS) do
        local cfg = config[ch]
        if cfg and cfg.x then
            if cfg.type == "bar" then
                raw_vals[ch] = sample_bar(data, stride, cfg)
            elseif cfg.type == "center-offset" then
                raw_vals[ch] = sample_center_offset(data, stride, cfg)
            elseif cfg.type == "digit" then
                raw_vals[ch] = sample_digit(data, stride, cfg.x, cfg.y, cfg.w, cfg.h)
            elseif cfg.type == "digits" then
                raw_vals[ch] = sample_digit(data, stride, cfg.x, cfg.y, cfg.w, cfg.h)
            end
        end
    end

    -- Smooth analog, snap digital
    for _, ch in ipairs({"throttle", "brake", "steering", "fuel"}) do
        if config[ch] then
            cur[ch] = cur[ch] * SMOOTH + (raw_vals[ch] or 0) * (1 - SMOOTH)
        end
    end
    for _, ch in ipairs({"gear", "speed"}) do
        if config[ch] then cur[ch] = raw_vals[ch] or 0 end
    end

    trace_idx = trace_idx + 1
    trace[trace_idx] = {
        throttle = cur.throttle, brake = cur.brake,
        steering = cur.steering, gear = cur.gear,
    }
    if trace_idx > TRACE_LEN * 2 then
        local t = {}
        for i = trace_idx - TRACE_LEN + 1, trace_idx do
            t[i - (trace_idx - TRACE_LEN)] = trace[i]
        end
        trace = t; trace_idx = TRACE_LEN
    end
end

-- ══════════════════════════════════════════════════════════════
-- ASS HELPERS
-- ══════════════════════════════════════════════════════════════

local ass_color = core.ass_color
local ass_bord_color = core.ass_bord_color
local ass_alpha = core.ass_alpha

-- ══════════════════════════════════════════════════════════════
-- TELEMETRY OVERLAY RENDERING
-- ══════════════════════════════════════════════════════════════

render_overlay = function()
    if not overlay_visible or cal_active then
        overlay.data = ""; overlay:update(); return
    end
    if not next(config) then return end -- no config loaded

    local osd_w, osd_h = mp.get_osd_size()
    if not osd_w or osd_w == 0 then return end
    overlay.res_x = osd_w
    overlay.res_y = osd_h

    local ww = osd_w * 0.30 * scale
    local wh = osd_h * 0.09 * scale
    local wx = widget_pos.x * osd_w - ww / 2
    local wy = widget_pos.y * osd_h - wh / 2
    wx = math.max(0, math.min(osd_w - ww, wx))
    wy = math.max(0, math.min(osd_h - wh, wy))

    -- Store for drag hit testing
    last_widget_rect = { x = wx, y = wy, w = ww, h = wh }

    local stripe_w = 4 * scale
    local gear_d = wh * 0.85
    local bar_w = math.max(6, 12 * scale)
    local bar_gap = math.max(2, 4 * scale)
    local has_gear = config.gear ~= nil
    local has_steering = config.steering ~= nil
    local has_brake = config.brake ~= nil
    local has_throttle = config.throttle ~= nil
    local gear_cx = wx + ww - gear_d / 2 - 6 * scale
    local gear_cy = wy + wh / 2
    local bars_right = gear_cx - gear_d / 2 - bar_gap
    if not has_gear and not has_steering then bars_right = wx + ww - 6 * scale end
    local bars_x = bars_right - bar_w * 2 - bar_gap
    if not has_brake or not has_throttle then
        bars_x = bars_right - bar_w
    end
    local trace_x = wx + stripe_w + 4 * scale
    local trace_w = math.max(10, bars_x - trace_x - 6 * scale)
    local trace_y, trace_h = wy + 2, wh - 4
    local bar_h = wh - 14 * scale
    local bar_y = wy + 12 * scale

    local ass = assdraw.ass_new()

    local function filled_rect(x1, y1, x2, y2, r, g, b, alpha)
        ass:new_event(); ass:pos(0, 0)
        ass:append(string.format("{\\bord0\\shad0%s%s\\p1}", ass_color(r, g, b), ass_alpha(alpha or 1)))
        ass:draw_start(); ass:rect_cw(x1, y1, x2, y2); ass:draw_stop()
    end

    -- Background
    filled_rect(wx, wy, wx + ww, wy + wh, 10, 10, 10, 0.82)
    filled_rect(wx, wy, wx + stripe_w, wy + wh, 225, 6, 0, 1)

    -- Grid
    for _, frac in ipairs({0.25, 0.5, 0.75}) do
        filled_rect(trace_x, trace_y + trace_h * (1 - frac), trace_x + trace_w, trace_y + trace_h * (1 - frac) + 1, 255, 255, 255, 0.06)
    end

    -- Traces
    if trace_idx >= 2 then
        local n = math.min(trace_idx, TRACE_LEN)
        local si = trace_idx - n + 1
        local step = trace_w / TRACE_LEN
        local function draw_trace(get_val, r, g, b, alpha)
            for i = 1, n - 1 do
                local e0, e1 = trace[si + i - 1], trace[si + i]
                if e0 and e1 then
                    local x0 = trace_x + (TRACE_LEN - n + i - 1) * step
                    local x1 = trace_x + (TRACE_LEN - n + i) * step
                    local y0 = trace_y + trace_h * (1 - get_val(e0))
                    local y1 = trace_y + trace_h * (1 - get_val(e1))
                    local th = 2 * scale
                    ass:new_event(); ass:pos(0, 0)
                    ass:append(string.format("{\\bord0\\shad0%s%s\\p1}", ass_color(r, g, b), ass_alpha(alpha or 0.85)))
                    ass:draw_start()
                    ass:move_to(x0, y0 - th/2); ass:line_to(x1, y1 - th/2)
                    ass:line_to(x1, y1 + th/2); ass:line_to(x0, y0 + th/2)
                    ass:draw_stop()
                end
            end
        end
        if has_gear then draw_trace(function(e) return (e.gear or 0) / 7 end, 255, 255, 255, 0.15) end
        if has_throttle then draw_trace(function(e) return e.throttle or 0 end, 34, 204, 68, 0.9) end
        if has_brake then draw_trace(function(e) return e.brake or 0 end, 221, 34, 34, 0.9) end
    end

    -- Pedal bars
    local bx = bars_x
    if has_brake then
        filled_rect(bx, bar_y, bx + bar_w, bar_y + bar_h, 221, 34, 34, 0.18)
        if cur.brake > 0.01 then
            local f = bar_h * cur.brake
            filled_rect(bx, bar_y + bar_h - f, bx + bar_w, bar_y + bar_h, 221, 34, 34, 1)
        end
        ass:new_event(); ass:pos(bx + bar_w / 2, bar_y - 2)
        ass:append(string.format("{\\an2\\bord0\\shad0%s\\fs%d\\fnmonospace\\b1}%d",
            ass_color(150,150,150), math.max(8, 9*scale), math.floor(cur.brake * 100)))
        bx = bx + bar_w + bar_gap
    end
    if has_throttle then
        filled_rect(bx, bar_y, bx + bar_w, bar_y + bar_h, 34, 204, 68, 0.18)
        if cur.throttle > 0.01 then
            local f = bar_h * cur.throttle
            filled_rect(bx, bar_y + bar_h - f, bx + bar_w, bar_y + bar_h, 34, 204, 68, 1)
        end
        ass:new_event(); ass:pos(bx + bar_w / 2, bar_y - 2)
        ass:append(string.format("{\\an2\\bord0\\shad0%s\\fs%d\\fnmonospace\\b1}%d",
            ass_color(150,150,150), math.max(8, 9*scale), math.floor(cur.throttle * 100)))
    end

    -- Gear circle
    if has_gear or has_steering then
        local gr = gear_d / 2 - 2
        local k = 0.5522847498 * gr
        ass:new_event(); ass:pos(0, 0)
        ass:append(string.format("{\\bord%.1f\\shad0%s%s%s\\p1}", math.max(2, 3*scale),
            ass_bord_color(68,68,68), ass_color(20,20,20), ass_alpha(1)))
        ass:draw_start()
        ass:move_to(gear_cx, gear_cy - gr)
        ass:bezier_curve(gear_cx+k, gear_cy-gr, gear_cx+gr, gear_cy-k, gear_cx+gr, gear_cy)
        ass:bezier_curve(gear_cx+gr, gear_cy+k, gear_cx+k, gear_cy+gr, gear_cx, gear_cy+gr)
        ass:bezier_curve(gear_cx-k, gear_cy+gr, gear_cx-gr, gear_cy+k, gear_cx-gr, gear_cy)
        ass:bezier_curve(gear_cx-gr, gear_cy-k, gear_cx-k, gear_cy-gr, gear_cx, gear_cy-gr)
        ass:draw_stop()

        if has_gear then
            ass:new_event(); ass:pos(gear_cx, gear_cy)
            ass:append(string.format("{\\an5\\bord0\\shad0%s\\fs%d\\fnmonospace\\b1}%d",
                ass_color(255,255,255), math.floor(gr * 1.1), cur.gear))
        end

        if has_steering then
            local ang = cur.steering * 270 * math.pi / 180
            local nr = gr + 3 * scale
            local nx = gear_cx + math.sin(ang) * nr
            local ny = gear_cy - math.cos(ang) * nr
            local ns = math.max(3, gr * 0.15)
            ass:new_event(); ass:pos(0, 0)
            ass:append(string.format("{\\bord0\\shad0%s\\p1}", ass_color(220,220,220)))
            ass:draw_start()
            ass:move_to(nx, ny - ns); ass:line_to(nx + ns*0.6, ny)
            ass:line_to(nx, ny + ns); ass:line_to(nx - ns*0.6, ny)
            ass:draw_stop()
        end
    end

    -- Config name label (small, bottom-right of widget)
    if config_name ~= "none" then
        ass:new_event(); ass:pos(wx + ww - 4, wy + wh - 2)
        ass:append(string.format("{\\an4\\bord0\\shad0%s%s\\fs%d\\fnmonospace}%s",
            ass_color(100,100,100), ass_alpha(0.5), math.max(6, 7*scale), config_name))
    end

    overlay.data = ass.text; overlay:update()
end

-- ══════════════════════════════════════════════════════════════
-- CALIBRATION OVERLAY
-- ══════════════════════════════════════════════════════════════

-- Compute the video rendering area within the OSD coordinate system.
-- Returns offset_x, offset_y, scale_x, scale_y so that:
--   osd_x = offset_x + video_x * scale_x
--   osd_y = offset_y + video_y * scale_y
local function get_video_transform()
    local osd_w, osd_h = mp.get_osd_size()
    local vid_w = mp.get_property_number("video-params/w", 1280)
    local vid_h = mp.get_property_number("video-params/h", 720)
    if not osd_w or osd_w == 0 then return 0, 0, 1, 1, vid_w, vid_h end

    local dar = mp.get_property_number("video-params/aspect", vid_w / vid_h)
    local osd_aspect = osd_w / osd_h
    local render_w, render_h, offset_x, offset_y
    if dar > osd_aspect then
        render_w = osd_w
        render_h = osd_w / dar
        offset_x = 0
        offset_y = (osd_h - render_h) / 2
    else
        render_h = osd_h
        render_w = osd_h * dar
        offset_x = (osd_w - render_w) / 2
        offset_y = 0
    end

    return offset_x, offset_y, render_w / vid_w, render_h / vid_h, vid_w, vid_h
end

-- Convert video pixel coords to OSD coords
local function video_to_osd(vx, vy)
    local ox, oy, sx, sy = get_video_transform()
    return ox + vx * sx, oy + vy * sy
end

-- Convert mouse position (OSD coords) to video pixel coords
local function mouse_to_video()
    local mx, my = mp.get_mouse_pos()
    local ox, oy, sx, sy, vid_w, vid_h = get_video_transform()

    local vx = (mx - ox) / sx
    local vy = (my - oy) / sy

    return math.floor(math.max(0, math.min(vid_w - 1, vx))),
           math.floor(math.max(0, math.min(vid_h - 1, vy)))
end

render_calibration = function()
    if not cal_active then
        cal_overlay.data = ""; cal_overlay:update(); return
    end

    -- Sync overlay resolution to current OSD size so coordinates match
    local osd_w, osd_h = mp.get_osd_size()
    if not osd_w or osd_w == 0 then return end
    cal_overlay.res_x = osd_w
    cal_overlay.res_y = osd_h

    local ass = assdraw.ass_new()

    local function filled_rect(x1, y1, x2, y2, r, g, b, alpha)
        ass:new_event(); ass:pos(0, 0)
        ass:append(string.format("{\\bord0\\shad0%s%s\\p1}", ass_color(r, g, b), ass_alpha(alpha or 1)))
        ass:draw_start(); ass:rect_cw(x1, y1, x2, y2); ass:draw_stop()
    end

    -- Semi-transparent background
    filled_rect(0, 0, osd_w, osd_h, 0, 0, 0, 0.3)

    -- Top bar
    filled_rect(0, 0, osd_w, 36, 15, 15, 15, 0.92)

    -- Title
    ass:new_event(); ass:pos(12, 18)
    ass:append(string.format("{\\an4\\bord0\\shad0%s\\fs14\\b1}⚙ CALIBRATE", ass_color(255,255,255)))

    -- Channel buttons
    local btn_x = 160
    for i, ch in ipairs(CHANNELS) do
        local is_sel = cal_channel == ch
        local is_cfg = config[ch] ~= nil
        local cr, cg, cb = 150, 150, 150
        if is_sel then cr, cg, cb = 225, 6, 0
        elseif is_cfg then cr, cg, cb = 34, 204, 68 end

        filled_rect(btn_x, 6, btn_x + 65, 30, cr, cg, cb, is_sel and 1 or 0.3)
        ass:new_event(); ass:pos(btn_x + 32, 18)
        ass:append(string.format("{\\an5\\bord0\\shad0%s\\fs11\\b1}%d:%s",
            ass_color(255,255,255), i, ch:sub(1,5):upper()))
        btn_x = btn_x + 72
    end

    -- Mode / status
    local status = ""
    if not cal_channel then
        status = "Press 1-6 to select a channel"
    elseif cal_mode == "draw" then
        status = cal_channel:upper() .. ": Draw a rectangle. [c]=color [m]=center [s]=save [Esc]=exit"
    elseif cal_mode == "pick-color" then
        status = cal_channel:upper() .. ": Click on the active/filled color"
    elseif cal_mode == "pick-center" then
        status = cal_channel:upper() .. ": Click the center/zero point"
    end
    ass:new_event(); ass:pos(btn_x + 16, 18)
    ass:append(string.format("{\\an4\\bord0\\shad0%s\\fs11}%s", ass_color(180,180,180), status))

    -- Config name
    ass:new_event(); ass:pos(osd_w - 10, 18)
    ass:append(string.format("{\\an6\\bord0\\shad0%s\\fs11}[%s]", ass_color(100,100,100), config_name))

    -- Draw existing regions (video coords → OSD coords via video_to_osd)
    for _, ch in ipairs(CHANNELS) do
        local cfg = config[ch]
        if cfg and cfg.x then
            local sx1, sy1 = video_to_osd(cfg.x, cfg.y)
            local sx2, sy2 = video_to_osd(cfg.x + cfg.w, cfg.y + cfg.h)
            local is_sel = cal_channel == ch
            local cr, cg, cb = 34, 204, 68
            if is_sel then cr, cg, cb = 225, 6, 0 end

            -- Border
            ass:new_event(); ass:pos(0, 0)
            ass:append(string.format("{\\bord%.1f\\shad0%s\\1a&HFF&\\p1}", is_sel and 2.5 or 1.5,
                ass_bord_color(cr, cg, cb)))
            ass:draw_start(); ass:rect_cw(sx1, sy1, sx2, sy2); ass:draw_stop()

            -- Label
            ass:new_event(); ass:pos(sx1 + 3, sy1 - 12)
            ass:append(string.format("{\\an1\\bord0\\shad1\\4c&H000000&%s\\fs10\\b1}%s",
                ass_color(cr, cg, cb), ch:upper()))

            -- Value preview
            if is_sel and cur[ch] then
                local val_str
                if cfg.type == "bar" then val_str = string.format("%.0f%%", (cur[ch] or 0) * 100)
                elseif cfg.type == "digit" or cfg.type == "digits" then val_str = tostring(math.floor(cur[ch] or 0))
                elseif cfg.type == "center-offset" then val_str = string.format("%.0f%%", (cur[ch] or 0) * 100)
                end
                if val_str then
                    ass:new_event(); ass:pos(sx2 + 8, (sy1 + sy2) / 2)
                    ass:append(string.format("{\\an4\\bord0\\shad1\\4c&H000000&%s\\fs14\\b1}→ %s",
                        ass_color(255, 220, 0), val_str))
                end
            end

            -- Center marker
            if cfg.type == "center-offset" and cfg.center_x then
                local cx_s, _ = video_to_osd(cfg.center_x, 0)
                ass:new_event(); ass:pos(0, 0)
                ass:append(string.format("{\\bord0\\shad0%s\\p1}", ass_color(255, 100, 255)))
                ass:draw_start()
                ass:rect_cw(cx_s - 1, sy1, cx_s + 1, sy2)
                ass:draw_stop()
            end
        end
    end

    -- Current draw rectangle (video coords → OSD)
    if cal_drawing and cal_draw_start and cal_draw_cur then
        local sx1, sy1 = video_to_osd(
            math.min(cal_draw_start.x, cal_draw_cur.x),
            math.min(cal_draw_start.y, cal_draw_cur.y))
        local sx2, sy2 = video_to_osd(
            math.max(cal_draw_start.x, cal_draw_cur.x),
            math.max(cal_draw_start.y, cal_draw_cur.y))
        ass:new_event(); ass:pos(0, 0)
        ass:append(string.format("{\\bord2\\shad0%s\\1a&HFF&\\p1}", ass_bord_color(225, 6, 0)))
        ass:draw_start(); ass:rect_cw(sx1, sy1, sx2, sy2); ass:draw_stop()

        -- Pixel dimensions (in video pixels)
        local pw = math.abs(cal_draw_cur.x - cal_draw_start.x)
        local ph = math.abs(cal_draw_cur.y - cal_draw_start.y)
        ass:new_event(); ass:pos(sx1 + 4, sy1 - 14)
        ass:append(string.format("{\\an1\\bord0\\shad1\\4c&H000000&%s\\fs11\\b1\\fnmonospace}%dx%d",
            ass_color(225, 6, 0), pw, ph))
    end

    -- ── Yellow cursor dot + debug info ──
    local mx, my = mp.get_mouse_pos()
    local dbg_vx, dbg_vy = mouse_to_video()

    -- Yellow dot at mouse position
    local dot_r = 4
    ass:new_event(); ass:pos(0, 0)
    ass:append(string.format("{\\bord1\\shad0\\3c&H000000&%s\\p1}", ass_color(255, 255, 0)))
    ass:draw_start()
    local k = 0.5522847498 * dot_r
    ass:move_to(mx, my - dot_r)
    ass:bezier_curve(mx+k, my-dot_r, mx+dot_r, my-k, mx+dot_r, my)
    ass:bezier_curve(mx+dot_r, my+k, mx+k, my+dot_r, mx, my+dot_r)
    ass:bezier_curve(mx-k, my+dot_r, mx-dot_r, my+k, mx-dot_r, my)
    ass:bezier_curve(mx-dot_r, my-k, mx-k, my-dot_r, mx, my-dot_r)
    ass:draw_stop()

    -- Crosshair lines through cursor
    ass:new_event(); ass:pos(0, 0)
    ass:append(string.format("{\\bord0\\shad0%s%s\\p1}", ass_color(255,255,0), ass_alpha(0.4)))
    ass:draw_start()
    ass:rect_cw(mx - 20, my, mx + 20, my + 1)
    ass:rect_cw(mx, my - 20, mx + 1, my + 20)
    ass:draw_stop()

    -- Debug text: video coords + pixel color, positioned next to cursor
    local dbg_info = string.format("(%d, %d)", dbg_vx, dbg_vy)
    if cal_last_screenshot then
        local pr, pg, pb = get_pixel(cal_last_screenshot.data, cal_last_screenshot.stride, dbg_vx, dbg_vy)
        dbg_info = dbg_info .. string.format("  rgb(%d,%d,%d)", pr, pg, pb)
    end
    ass:new_event(); ass:pos(mx + 12, my - 8)
    ass:append(string.format("{\\an1\\bord0\\shad1\\4c&H000000&%s\\fs11\\fnmonospace\\b1}%s",
        ass_color(255, 255, 0), dbg_info))

    -- Also show in top-right corner for visibility
    ass:new_event(); ass:pos(osd_w - 10, 18)
    ass:append(string.format("{\\an6\\bord0\\shad0%s\\fs11\\fnmonospace}video: %d,%d  [%s]",
        ass_color(100,100,100), dbg_vx, dbg_vy, config_name))

    cal_overlay.data = ass.text; cal_overlay:update()
end

-- ══════════════════════════════════════════════════════════════
-- CALIBRATION MODE
-- ══════════════════════════════════════════════════════════════

local cal_bindings_active = false

enter_calibration = function()
    cal_active = true
    cal_channel = nil
    cal_channel_idx = 0
    cal_mode = "draw"
    mp.set_property_bool("pause", true)

    -- Take a frame for live preview
    pcall(process_frame)

    if not cal_bindings_active then
        cal_bindings_active = true

        -- Channel selection: 1-6
        for i, ch in ipairs(CHANNELS) do
            mp.add_forced_key_binding(tostring(i), "cal-ch-" .. i, function()
                if not cal_active then return end
                cal_channel = ch
                cal_channel_idx = i
                cal_mode = "draw"
                mp.osd_message(ch:upper() .. " selected — draw a rectangle")
                render_calibration()
            end)
        end

        -- Draw with mouse (two-click: no drag needed)
        mp.add_forced_key_binding("MBTN_LEFT", "cal-click", function()
            if not cal_active or not cal_channel then return end

            do
                local mx, my = mp.get_mouse_pos()
                -- Skip if in toolbar area
                if my < 36 then return end

                if cal_mode == "pick-color" then
                    local vx, vy = mouse_to_video()
                    if cal_last_screenshot then
                        local r, g, b = get_pixel(cal_last_screenshot.data, cal_last_screenshot.stride, vx, vy)
                        local cfg = config[cal_channel] or {}
                        -- Auto-detect which channel is dominant
                        cfg.active_r = r; cfg.active_g = g; cfg.active_b = b
                        cfg.color_dist = cfg.color_dist or 60
                        config[cal_channel] = cfg
                        mp.osd_message(string.format("Active color set: rgb(%d,%d,%d)", r, g, b))
                    end
                    cal_mode = "draw"
                    render_calibration()
                    return
                end

                if cal_mode == "pick-center" then
                    local vx, vy = mouse_to_video()
                    local cfg = config[cal_channel]
                    if cfg then cfg.center_x = vx end
                    mp.osd_message(string.format("Center set at x=%d", vx))
                    cal_mode = "draw"
                    render_calibration()
                    return
                end

                -- Two-click rectangle: first click = corner 1, second click = corner 2
                local vx, vy = mouse_to_video()

                if not cal_drawing then
                    -- First click: set corner 1, start tracking
                    cal_drawing = true
                    cal_draw_start = { x = vx, y = vy }
                    cal_draw_cur = { x = vx, y = vy }
                    mp.osd_message("Corner 1 set — move to opposite corner and click")
                else
                    -- Second click: set corner 2, finalize rect
                    cal_draw_cur = { x = vx, y = vy }
                    local x1 = math.min(cal_draw_start.x, cal_draw_cur.x)
                    local y1 = math.min(cal_draw_start.y, cal_draw_cur.y)
                    local x2 = math.max(cal_draw_start.x, cal_draw_cur.x)
                    local y2 = math.max(cal_draw_start.y, cal_draw_cur.y)
                    local w = x2 - x1; local h = y2 - y1

                    if w > 3 and h > 3 then
                        local defaults = CHANNEL_DEFAULTS[cal_channel] or { type = "bar" }
                        local cfg = config[cal_channel] or {}
                        for k, v in pairs(defaults) do
                            if cfg[k] == nil then cfg[k] = v end
                        end
                        cfg.x = x1; cfg.y = y1; cfg.w = w; cfg.h = h
                        if cfg.type == "center-offset" and not cfg.center_x then
                            cfg.center_x = x1 + w / 2
                        end
                        config[cal_channel] = cfg
                        mp.osd_message(string.format("%s: %dx%d at (%d,%d)",
                            cal_channel:upper(), w, h, x1, y1))
                    else
                        mp.osd_message("Too small — try again")
                    end

                    cal_drawing = false
                    cal_draw_start = nil; cal_draw_cur = nil
                    render_calibration()
                end
            end
        end)

        -- Mouse move polling is handled in on_tick (MOUSE_MOVE binding doesn't exist in mpv)

        -- Pick color mode
        mp.add_forced_key_binding("c", "cal-pick-color", function()
            if not cal_active or not cal_channel then return end
            cal_mode = "pick-color"
            mp.osd_message("Click on the active/filled color")
            render_calibration()
        end)

        -- Pick center mode
        mp.add_forced_key_binding("m", "cal-pick-center", function()
            if not cal_active or not cal_channel then return end
            cal_mode = "pick-center"
            mp.osd_message("Click the center/zero point")
            render_calibration()
        end)

        -- Save
        mp.add_forced_key_binding("s", "cal-save", function()
            if not cal_active then return end
            if not next(config) then
                mp.osd_message("Nothing to save — draw some regions first")
                return
            end
            -- Generate name from timestamp
            local name = os.date("cal-%Y%m%d-%H%M%S")
            save_config(name)
            mp.osd_message("Saved: " .. name)
        end)

        -- Exit
        mp.add_forced_key_binding("ESC", "cal-exit", function() exit_calibration() end)
        mp.add_forced_key_binding("SPACE", "cal-exit2", function() exit_calibration() end)
    end

    render_calibration()
    mp.osd_message("Calibration mode — press 1-6 to select channel, draw rectangles")
end

exit_calibration = function()
    cal_active = false
    cal_drawing = false
    cal_overlay.data = ""; cal_overlay:update()

    -- Remove calibration bindings
    if cal_bindings_active then
        for i = 1, 6 do mp.remove_key_binding("cal-ch-" .. i) end
        mp.remove_key_binding("cal-click")
        mp.remove_key_binding("cal-pick-color")
        mp.remove_key_binding("cal-pick-center")
        mp.remove_key_binding("cal-save")
        mp.remove_key_binding("cal-exit")
        mp.remove_key_binding("cal-exit2")
        cal_bindings_active = false
    end

    mp.set_property_bool("pause", false)
    mp.osd_message("Calibration off")
    render_overlay()
end

-- ══════════════════════════════════════════════════════════════
-- TIMER
-- ══════════════════════════════════════════════════════════════

local frame_failures = 0

local function on_tick()
    if dragging then
        local mx, my = mp.get_mouse_pos()
        local osd_w, osd_h = mp.get_osd_size()
        if osd_w and osd_w > 0 then
            widget_pos.x = math.max(0.05, math.min(0.95, mx / osd_w + drag_offset.x))
            widget_pos.y = math.max(0.05, math.min(0.95, my / osd_h + drag_offset.y))
        end
    end

    if not dragging and not cal_active then
        local ok, err = pcall(process_frame)
        if not ok then
            frame_failures = frame_failures + 1
            if frame_failures <= 3 then msg.verbose("frame: " .. tostring(err)) end
        else frame_failures = 0 end
    end

    if cal_active then
        -- Poll mouse position for rectangle drawing
        if cal_drawing then
            local vx, vy = mouse_to_video()
            cal_draw_cur = { x = vx, y = vy }
        end
        -- In calibration, still sample for live preview
        pcall(process_frame)
        pcall(render_calibration)
    else
        pcall(render_overlay)
    end
end

local function start_sampling()
    if timer then timer:kill() end
    timer = mp.add_periodic_timer(0.1, on_tick)
end

local function stop_sampling()
    if timer then timer:kill(); timer = nil end
    overlay.data = ""; overlay:update()
    cal_overlay.data = ""; cal_overlay:update()
end

-- ══════════════════════════════════════════════════════════════
-- KEY BINDINGS
-- ══════════════════════════════════════════════════════════════

mp.add_key_binding("ctrl+t", "toggle-telemetry", function()
    overlay_visible = not overlay_visible
    if overlay_visible then start_sampling(); mp.osd_message("Telemetry ON")
    else stop_sampling(); mp.osd_message("Telemetry OFF") end
end)

mp.add_key_binding("ctrl+c", "toggle-calibration", function()
    if cal_active then exit_calibration() else enter_calibration() end
end)

mp.add_key_binding("ctrl+n", "cycle-config", function()
    all_config_names = list_configs()
    if #all_config_names == 0 then
        mp.osd_message("No saved calibrations"); return
    end
    config_idx = config_idx % #all_config_names + 1
    load_config(all_config_names[config_idx])
    mp.osd_message("Config: " .. config_name .. " (" .. config_idx .. "/" .. #all_config_names .. ")")
end)

local presets = {
    {x=0.50, y=0.92}, {x=0.17, y=0.92}, {x=0.83, y=0.92}, {x=0.50, y=0.06},
}
local preset_idx = 1
mp.add_key_binding("ctrl+g", "cycle-telemetry-pos", function()
    preset_idx = preset_idx % #presets + 1
    widget_pos.x = presets[preset_idx].x; widget_pos.y = presets[preset_idx].y
end)

mp.add_key_binding("MBTN_LEFT", "telemetry-drag", function(ev)
    if cal_active then return end
    if ev.event == "down" then
        local mx, my = mp.get_mouse_pos()
        local r = last_widget_rect
        if mx >= r.x and mx <= r.x + r.w and my >= r.y and my <= r.y + r.h then
            dragging = true
            local osd_w, osd_h = mp.get_osd_size()
            if osd_w and osd_w > 0 then
                drag_offset.x = widget_pos.x - mx / osd_w
                drag_offset.y = widget_pos.y - my / osd_h
            end
        end
    elseif ev.event == "up" then dragging = false end
end, { complex = true })

mp.add_key_binding("ctrl+=", "telemetry-bigger", function()
    scale = math.min(2.0, scale + 0.1)
    mp.osd_message(string.format("Scale: %.0f%%", scale * 100))
end)
mp.add_key_binding("ctrl+-", "telemetry-smaller", function()
    scale = math.max(0.5, scale - 0.1)
    mp.osd_message(string.format("Scale: %.0f%%", scale * 100))
end)

-- ══════════════════════════════════════════════════════════════
-- EVENTS
-- ══════════════════════════════════════════════════════════════

mp.register_event("file-loaded", function()
    msg.info("File loaded")
    if not load_most_recent() then
        msg.info("No calibration found — Ctrl+C to calibrate")
    end
    if overlay_visible then start_sampling() end
end)

mp.register_event("end-file", stop_sampling)

mp.observe_property("vo-configured", "bool", function(_, configured)
    if configured and overlay_visible then
        overlay = mp.create_osd_overlay("ass-events")
        cal_overlay = mp.create_osd_overlay("ass-events")
    end
end)

mp.observe_property("fullscreen", "bool", function()
    if overlay_visible then mp.add_timeout(0.2, render_overlay) end
end)

-- Load on startup
load_most_recent()
msg.info("Racing Telemetry loaded. Ctrl+T=toggle Ctrl+C=calibrate Ctrl+N=cycle configs Ctrl+G=position Ctrl+±=size")
