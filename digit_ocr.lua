-- digit_ocr.lua — ONNX digit recognition via pure LuaJIT FFI (no C bridge needed)
--
-- Calls onnxruntime's C API directly through FFI function pointer table.
-- Function indices derived from onnxruntime 1.24.x onnxruntime_c_api.h.

local ffi = require("ffi")
local msg = require("mp.msg")

local M = {}

ffi.cdef[[
    typedef struct OrtEnv OrtEnv;
    typedef struct OrtSession OrtSession;
    typedef struct OrtSessionOptions OrtSessionOptions;
    typedef struct OrtRunOptions OrtRunOptions;
    typedef struct OrtMemoryInfo OrtMemoryInfo;
    typedef struct OrtValue OrtValue;
    typedef struct OrtStatus OrtStatus;

    // OrtApiBase
    typedef struct {
        const void* (*GetApi)(uint32_t version);
        const char* (*GetVersionString)(void);
    } OrtApiBase;

    const OrtApiBase* OrtGetApiBase(void);
]]

local CLASSES = {"0","1","2","3","4","5","6","7","8","9"}
local INPUT_SIZE = 32

-- OrtApi function pointer indices (from onnxruntime_c_api.h 1.24.x struct OrtApi)
-- Each entry in the struct is a function pointer (8 bytes on 64-bit).
-- Indices probed from onnxruntime 1.24.4 via offsetof (see probe_api.c)
local IDX = {
    CreateStatus                  = 0,
    GetErrorCode                  = 1,
    GetErrorMessage               = 2,
    CreateEnv                     = 3,
    CreateSession                 = 7,
    Run                           = 9,
    CreateSessionOptions          = 10,
    SetIntraOpNumThreads          = 24,
    CreateTensorWithDataAsOrtValue = 49,
    GetTensorMutableData          = 51,
    CreateCpuMemoryInfo           = 69,
    ReleaseEnv                    = 92,
    ReleaseStatus                 = 93,
    ReleaseMemoryInfo             = 94,
    ReleaseSession                = 95,
    ReleaseValue                  = 96,
    ReleaseSessionOptions         = 100,
}

local ort_lib = nil
local api = nil    -- pointer to function pointer array
local env = nil
local session = nil
local mem_info = nil
local initialized = false

local IS_WINDOWS = ffi.os == "Windows"
local PATH_SEP = IS_WINDOWS and "\\" or "/"

local function path_join(...)
    local parts = {...}
    local out = table.concat(parts, PATH_SEP)
    return out:gsub("[\\/]+", PATH_SEP)
end

local function file_exists(path)
    local f = io.open(path, "rb")
    if f then f:close(); return true end
    return false
end

local function dirname(path)
    return path:match("^(.+)[\\/][^\\/]+$") or "."
end

-- ONNX Runtime's C API uses wchar_t paths on Windows. LuaJIT strings are
-- UTF-8, so build a UTF-16LE buffer for CreateSession.
local function utf8_to_utf16le(str)
    local codepoints = {}
    local i, n = 1, #str
    while i <= n do
        local b1 = str:byte(i)
        local cp
        if b1 < 0x80 then
            cp = b1; i = i + 1
        elseif b1 < 0xE0 then
            local b2 = str:byte(i + 1) or 0
            cp = (b1 % 0x20) * 0x40 + (b2 % 0x40); i = i + 2
        elseif b1 < 0xF0 then
            local b2, b3 = str:byte(i + 1) or 0, str:byte(i + 2) or 0
            cp = (b1 % 0x10) * 0x1000 + (b2 % 0x40) * 0x40 + (b3 % 0x40); i = i + 3
        else
            local b2, b3, b4 = str:byte(i + 1) or 0, str:byte(i + 2) or 0, str:byte(i + 3) or 0
            cp = (b1 % 0x08) * 0x40000 + (b2 % 0x40) * 0x1000 + (b3 % 0x40) * 0x40 + (b4 % 0x40); i = i + 4
        end
        if cp <= 0xFFFF then
            codepoints[#codepoints + 1] = cp
        else
            cp = cp - 0x10000
            codepoints[#codepoints + 1] = 0xD800 + math.floor(cp / 0x400)
            codepoints[#codepoints + 1] = 0xDC00 + (cp % 0x400)
        end
    end
    local buf = ffi.new("uint16_t[?]", #codepoints + 1)
    for j, cp in ipairs(codepoints) do buf[j - 1] = cp end
    buf[#codepoints] = 0
    return buf
end

-- Call an OrtApi function by index, casting to the given type
local function api_fn(idx, ctype)
    local fp = ffi.cast("void**", api)
    return ffi.cast(ctype, fp[idx])
end

local function check_status(status)
    if status ~= nil then
        local GetErrorMessage = api_fn(IDX.GetErrorMessage, "const char*(*)(const OrtStatus*)")
        local errmsg = ffi.string(GetErrorMessage(status))
        local ReleaseStatus = api_fn(IDX.ReleaseStatus, "void(*)(OrtStatus*)")
        ReleaseStatus(status)
        error("ORT: " .. errmsg)
    end
end

function M.init(model_path, script_dir)
    if initialized then return true end

    script_dir = script_dir or dirname(model_path)
    local bundled_win_dir = path_join(script_dir, "third_party", "onnxruntime", "win-x64")

    -- Load onnxruntime. On Windows we bundle the CPU DLLs so users do not need
    -- to install ONNX Runtime separately; elsewhere we prefer the system copy.
    local paths
    if IS_WINDOWS then
        local bundled_providers = path_join(bundled_win_dir, "onnxruntime_providers_shared.dll")
        if file_exists(bundled_providers) then pcall(ffi.load, bundled_providers) end
        paths = {
            path_join(bundled_win_dir, "onnxruntime.dll"),
            path_join(script_dir, "onnxruntime.dll"),
            path_join(script_dir, "bin", "onnxruntime.dll"),
            "onnxruntime",
            "onnxruntime.dll",
        }
    else
        paths = {
            "libonnxruntime",                              -- system LD path (works everywhere if installed)
            "/opt/homebrew/lib/libonnxruntime.dylib",       -- macOS ARM (Homebrew)
            "/usr/local/lib/libonnxruntime.dylib",          -- macOS Intel (Homebrew)
            "/usr/lib/libonnxruntime.so",                   -- Linux system
            "/usr/local/lib/libonnxruntime.so",             -- Linux /usr/local
        }
    end
    for _, p in ipairs(paths) do
        local ok, lib = pcall(ffi.load, p)
        if ok then ort_lib = lib; break end
    end
    if not ort_lib then
        msg.warn("digit_ocr: could not load libonnxruntime")
        return false
    end

    local base = ort_lib.OrtGetApiBase()
    api = base.GetApi(18)  -- ORT API version 18
    if api == nil then
        -- Try older versions
        for v = 17, 13, -1 do
            api = base.GetApi(v)
            if api ~= nil then break end
        end
    end
    if api == nil then
        msg.error("digit_ocr: could not get OrtApi")
        return false
    end

    msg.info("digit_ocr: ORT version " .. ffi.string(base.GetVersionString()))

    local ok, err = pcall(function()
        -- CreateEnv
        local CreateEnv = api_fn(IDX.CreateEnv,
            "OrtStatus*(*)(int, const char*, OrtEnv**)")
        local env_p = ffi.new("OrtEnv*[1]")
        check_status(CreateEnv(2, "digit_ocr", env_p))
        env = env_p[0]

        -- CreateSessionOptions
        local CreateSessionOptions = api_fn(IDX.CreateSessionOptions,
            "OrtStatus*(*)(OrtSessionOptions**)")
        local opts_p = ffi.new("OrtSessionOptions*[1]")
        check_status(CreateSessionOptions(opts_p))
        local opts = opts_p[0]

        -- SetIntraOpNumThreads(1)
        local SetThreads = api_fn(IDX.SetIntraOpNumThreads,
            "OrtStatus*(*)(OrtSessionOptions*, int)")
        SetThreads(opts, 1)

        -- CreateSession. ORTCHAR_T is wchar_t on Windows and char elsewhere.
        local sess_p = ffi.new("OrtSession*[1]")
        if IS_WINDOWS then
            local CreateSession = api_fn(IDX.CreateSession,
                "OrtStatus*(*)(OrtEnv*, const wchar_t*, OrtSessionOptions*, OrtSession**)")
            local wide_model_path = utf8_to_utf16le(model_path)
            check_status(CreateSession(env, ffi.cast("const wchar_t*", wide_model_path), opts, sess_p))
        else
            local CreateSession = api_fn(IDX.CreateSession,
                "OrtStatus*(*)(OrtEnv*, const char*, OrtSessionOptions*, OrtSession**)")
            check_status(CreateSession(env, model_path, opts, sess_p))
        end
        session = sess_p[0]

        -- ReleaseSessionOptions
        local ReleaseOpts = api_fn(IDX.ReleaseSessionOptions, "void(*)(OrtSessionOptions*)")
        ReleaseOpts(opts)

        -- CreateCpuMemoryInfo
        local CreateCpuMem = api_fn(IDX.CreateCpuMemoryInfo,
            "OrtStatus*(*)(int, int, OrtMemoryInfo**)")
        local mi_p = ffi.new("OrtMemoryInfo*[1]")
        check_status(CreateCpuMem(0, 0, mi_p))
        mem_info = mi_p[0]
    end)

    if not ok then
        msg.error("digit_ocr: init failed: " .. tostring(err))
        return false
    end

    initialized = true
    msg.info("digit_ocr: ONNX model loaded")
    return true
end

function M.classify(data, stride, x, y, w, h)
    if not initialized then return nil end

    -- Preprocess: grayscale 32x32, normalized to [-1,1]
    local input = ffi.new("float[?]", INPUT_SIZE * INPUT_SIZE)
    for oy = 0, INPUT_SIZE - 1 do
        for ox = 0, INPUT_SIZE - 1 do
            local sx = x + math.floor(ox * w / INPUT_SIZE)
            local sy = y + math.floor(oy * h / INPUT_SIZE)
            local offset = sy * stride + sx * 4
            if offset >= 0 and offset + 3 <= #data then
                local b = data:byte(offset + 1)
                local g = data:byte(offset + 2)
                local r = data:byte(offset + 3)
                local gray = (r * 0.299 + g * 0.587 + b * 0.114) / 255.0
                input[oy * INPUT_SIZE + ox] = gray * 2 - 1
            else
                input[oy * INPUT_SIZE + ox] = -1
            end
        end
    end

    local CreateTensor = api_fn(IDX.CreateTensorWithDataAsOrtValue,
        "OrtStatus*(*)(OrtMemoryInfo*, void*, size_t, const int64_t*, size_t, int, OrtValue**)")
    local Run = api_fn(IDX.Run,
        "OrtStatus*(*)(OrtSession*, OrtRunOptions*, const char* const*, const OrtValue* const*, size_t, const char* const*, size_t, OrtValue**)")
    local GetData = api_fn(IDX.GetTensorMutableData,
        "OrtStatus*(*)(OrtValue*, void**)")
    local ReleaseValue = api_fn(IDX.ReleaseValue, "void(*)(OrtValue*)")

    -- Create input tensor
    local shape = ffi.new("int64_t[4]", {1, 1, INPUT_SIZE, INPUT_SIZE})
    local tensor_p = ffi.new("OrtValue*[1]")
    local status = CreateTensor(mem_info, input, INPUT_SIZE * INPUT_SIZE * 4, shape, 4, 1, tensor_p)
    if status ~= nil then return nil end

    -- Run inference
    local input_names = ffi.new("const char*[1]", {"image"})
    local output_names = ffi.new("const char*[1]", {"logits"})
    local output_p = ffi.new("OrtValue*[1]")

    status = Run(session, nil, input_names, ffi.cast("const OrtValue* const*", tensor_p), 1, output_names, 1, output_p)
    ReleaseValue(tensor_p[0])
    if status ~= nil then return nil end

    -- Get logits
    local data_p = ffi.new("void*[1]")
    GetData(output_p[0], data_p)
    local logits = ffi.cast("float*", data_p[0])

    -- Argmax + confidence
    local max_idx, max_val = 0, logits[0]
    for i = 1, #CLASSES - 1 do
        if logits[i] > max_val then max_val = logits[i]; max_idx = i end
    end
    local exp_sum = 0
    for i = 0, #CLASSES - 1 do
        exp_sum = exp_sum + math.exp(logits[i] - max_val)
    end
    local confidence = 1.0 / exp_sum

    ReleaseValue(output_p[0])

    if confidence > 0.3 then
        return CLASSES[max_idx + 1], confidence
    end
    return nil, confidence
end

function M.cleanup()
    if session then api_fn(IDX.ReleaseSession, "void(*)(OrtSession*)")(session); session = nil end
    if mem_info then api_fn(IDX.ReleaseMemoryInfo, "void(*)(OrtMemoryInfo*)")(mem_info); mem_info = nil end
    if env then api_fn(IDX.ReleaseEnv, "void(*)(OrtEnv*)")(env); env = nil end
    initialized = false
end

return M
