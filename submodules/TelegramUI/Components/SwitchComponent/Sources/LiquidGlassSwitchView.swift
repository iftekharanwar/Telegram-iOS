import Foundation
import UIKit
import MetalKit

final class LiquidGlassSwitchView: UIControl {
    // MARK: - Public Properties
    var isOn: Bool = false {
        didSet {
            if isOn != oldValue {
                animateStateChange(to: isOn)
            }
        }
    }

    var valueChanged: ((Bool) -> Void)?

    // MARK: - Private Properties
    private let trackWidth: CGFloat = 64.0
    private let trackHeight: CGFloat = 31.0
    private let thumbWidth: CGFloat = 34.0
    private let thumbHeight: CGFloat = 27.0

    private var trackView: UIView!
    private var trackTintLayer: CALayer!
    private var trackInnerShadow: CALayer!

    private var glassTrack: GlassTrackView?

    private var solidThumb: UIView!
    private var glassThumb: GlassThumbView?

    private var thumbPosition: CGFloat = 0.0
    private var targetPosition: CGFloat = 0.0
    private var velocity: CGFloat = 0.0
    private var displayLink: CADisplayLink?
    private var isInteracting: Bool = false
    private var touchScale: CGFloat = 1.0

    private let springStiffness: CGFloat = 320.0
    private let springDamping: CGFloat = 18.0

    // MARK: - Initialization
    override init(frame: CGRect) {
        super.init(frame: CGRect(x: 0, y: 0, width: trackWidth, height: trackHeight))
        setupViews()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupViews()
    }

    private func setupViews() {
        // Track background
        trackView = UIView()
        trackView.layer.cornerRadius = trackHeight / 2
        trackView.backgroundColor = UIColor(white: 0.9, alpha: 1.0)
        trackView.layer.shadowColor = UIColor.black.cgColor
        trackView.layer.shadowOpacity = 0.1
        trackView.layer.shadowOffset = CGSize(width: 0, height: 1)
        trackView.layer.shadowRadius = 2
        addSubview(trackView)

        // Inner shadow for depth
        trackInnerShadow = CALayer()
        trackInnerShadow.frame = trackView.bounds
        trackInnerShadow.backgroundColor = UIColor.clear.cgColor
        trackInnerShadow.shadowColor = UIColor.black.cgColor
        trackInnerShadow.shadowOffset = CGSize(width: 0, height: 1)
        trackInnerShadow.shadowOpacity = 0.2
        trackInnerShadow.shadowRadius = 2.0
        trackInnerShadow.cornerRadius = trackHeight / 2
        trackInnerShadow.masksToBounds = false
        trackView.layer.addSublayer(trackInnerShadow)

        // Tint layer for color morph
        trackTintLayer = CALayer()
        trackTintLayer.backgroundColor = UIColor.systemGreen.cgColor
        trackTintLayer.cornerRadius = trackHeight / 2
        trackTintLayer.opacity = 0.0
        trackView.layer.addSublayer(trackTintLayer)

        // Solid thumb (oval)
        solidThumb = UIView()
        solidThumb.backgroundColor = .white
        solidThumb.isOpaque = false
        solidThumb.layer.cornerRadius = thumbHeight / 2
        solidThumb.layer.shadowColor = UIColor.black.cgColor
        solidThumb.layer.shadowOpacity = 0.2
        solidThumb.layer.shadowOffset = CGSize(width: 0, height: 2)
        solidThumb.layer.shadowRadius = 4
        solidThumb.isUserInteractionEnabled = false
        addSubview(solidThumb)

        // Gestures
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap))
        addGestureRecognizer(tapGesture)

        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        addGestureRecognizer(panGesture)

        // Start physics loop
        startPhysicsLoop()

        // Initial layout
        updateLayout()
    }

    override var intrinsicContentSize: CGSize {
        return CGSize(width: trackWidth, height: trackHeight)
    }

    override func sizeThatFits(_ size: CGSize) -> CGSize {
        return CGSize(width: trackWidth, height: trackHeight)
    }

    override func layoutSubviews() {
        super.layoutSubviews()

        trackView.frame = bounds
        trackInnerShadow.frame = trackView.bounds
        trackTintLayer.frame = trackView.bounds
        glassTrack?.frame = bounds

        updateLayout()
    }

    private func updateLayout() {
        let trackRange = trackWidth - thumbWidth - 4.0
        let thumbX = 2.0 + (trackRange * thumbPosition)

        let scaledThumbWidth = thumbWidth * touchScale
        let scaledThumbHeight = thumbHeight * touchScale
        let thumbFrame = CGRect(
            x: thumbX + (thumbWidth - scaledThumbWidth) / 2,
            y: (trackHeight - scaledThumbHeight) / 2,
            width: scaledThumbWidth,
            height: scaledThumbHeight
        )

        solidThumb.frame = thumbFrame
        solidThumb.layer.cornerRadius = scaledThumbHeight / 2
        glassThumb?.frame = thumbFrame
    }

    // MARK: - Morph Transitions
    private func animateThumbScale(to scale: CGFloat, duration: TimeInterval, overshoot: Bool = false) {
        if overshoot {
            UIView.animate(withDuration: duration, delay: 0, usingSpringWithDamping: 0.7, initialSpringVelocity: 0.3, options: [.curveEaseOut, .allowUserInteraction]) {
                self.touchScale = scale
                self.updateLayout()
            }
        } else {
            UIView.animate(withDuration: duration, delay: 0, options: [.curveEaseOut, .allowUserInteraction]) {
                self.touchScale = scale
                self.updateLayout()
            }
        }
    }

    private func morphToGlass() {
        if glassThumb == nil {
            glassThumb = GlassThumbView()
            glassThumb!.alpha = 0
            glassThumb!.isUserInteractionEnabled = false
            insertSubview(glassThumb!, aboveSubview: solidThumb)
            updateLayout()
        }

        UIView.animate(withDuration: 0.2, delay: 0, options: [.curveEaseInOut, .allowUserInteraction]) {
            self.solidThumb.alpha = 0
            self.solidThumb.layer.shadowOpacity = 0
            self.glassThumb?.alpha = 1.0
        }

        glassThumb?.startRendering()
        glassThumb?.setPressed(true)
    }

    private func morphToSolid() {
        UIView.animate(withDuration: 0.35, delay: 0, options: [.curveEaseInOut, .allowUserInteraction]) {
            self.solidThumb.alpha = 1.0
            self.solidThumb.layer.shadowOpacity = 0.2
            self.glassThumb?.alpha = 0
        } completion: { _ in
            self.glassThumb?.stopRendering()
        }

        glassThumb?.setPressed(false)
    }

    // MARK: - Physics Loop
    private func startPhysicsLoop() {
        let link = CADisplayLink(target: self, selector: #selector(updatePhysics))
        link.preferredFramesPerSecond = 120
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    @objc private func updatePhysics() {
        let dt: CGFloat = 1.0 / 120.0

        let displacement = targetPosition - thumbPosition
        let springForce = springStiffness * displacement
        let dampingForce = -springDamping * velocity
        let acceleration = springForce + dampingForce

        velocity += acceleration * dt
        thumbPosition += velocity * dt

        let overshootAmount: CGFloat = 0.03
        if thumbPosition < -overshootAmount {
            thumbPosition = -overshootAmount
            velocity = abs(velocity) * 0.3
        } else if thumbPosition > 1.0 + overshootAmount {
            thumbPosition = 1.0 + overshootAmount
            velocity = -abs(velocity) * 0.3
        }

        if abs(velocity) < 0.01 && abs(displacement) < 0.01 {
            thumbPosition = max(0, min(1, thumbPosition))
        }

        if abs(displacement) > 0.0001 || abs(velocity) > 0.0001 {
            updateLayout()
        }
    }

    // MARK: - Touch Handling
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesBegan(touches, with: event)

        isInteracting = true
        morphToGlass()
        animateThumbScale(to: 1.4, duration: 0.25)
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesEnded(touches, with: event)

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self = self else { return }
            if !self.isInteracting {
                self.morphToSolid()
                self.animateThumbScale(to: 1.0, duration: 0.4, overshoot: true)
            }
        }
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesCancelled(touches, with: event)

        if !isInteracting {
            morphToSolid()
            animateThumbScale(to: 1.0, duration: 0.4, overshoot: true)
        }
    }

    // MARK: - Gesture Handlers
    @objc private func handleTap() {
        toggle()

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
            self?.isInteracting = false
        }
    }

    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        switch gesture.state {
        case .began:
            isInteracting = true

        case .changed:
            let translation = gesture.translation(in: self)
            if abs(translation.x) > 10 {
                let shouldBeOn = translation.x > 0
                if shouldBeOn != isOn {
                    toggle()
                }
            }

        case .ended, .cancelled:
            isInteracting = false
            morphToSolid()
            animateThumbScale(to: 1.0, duration: 0.4, overshoot: true)

        default:
            break
        }
    }

    func toggle() {
        isOn.toggle()
        valueChanged?(isOn)
        sendActions(for: .valueChanged)
    }

    // MARK: - Animation
    private func animateStateChange(to newState: Bool) {
        targetPosition = newState ? 1.0 : 0.0

        velocity = newState ? 8.0 : -8.0

        morphTrackToGlass()

        UIView.animate(withDuration: 0.1, delay: 0, options: [.curveEaseOut]) {
            self.touchScale = 0.92
        } completion: { _ in
            UIView.animate(withDuration: 0.25, delay: 0, usingSpringWithDamping: 0.65, initialSpringVelocity: 0.5, options: [.curveEaseOut]) {
                self.touchScale = 1.0
                self.updateLayout()
            }
        }

        CATransaction.begin()
        CATransaction.setAnimationDuration(0.3)
        CATransaction.setAnimationTimingFunction(CAMediaTimingFunction(name: .easeOut))

        trackTintLayer.opacity = newState ? 1.0 : 0.0
        trackView.backgroundColor = newState ? UIColor.systemGreen : UIColor(white: 0.9, alpha: 1.0)

        CATransaction.commit()

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
            self?.morphTrackToSolid()
        }
    }

    private func morphTrackToGlass() {
        if glassTrack == nil {
            glassTrack = GlassTrackView()
            glassTrack!.alpha = 0
            glassTrack!.isUserInteractionEnabled = false
            glassTrack!.frame = trackView.frame
            insertSubview(glassTrack!, aboveSubview: trackView)
        }

        UIView.animate(withDuration: 0.2, delay: 0, options: [.curveEaseInOut, .allowUserInteraction]) {
            self.glassTrack?.alpha = 1.0
        }

        glassTrack?.startRendering()
    }

    private func morphTrackToSolid() {
        UIView.animate(withDuration: 0.3, delay: 0, options: [.curveEaseInOut, .allowUserInteraction]) {
            self.glassTrack?.alpha = 0
        } completion: { _ in
            self.glassTrack?.stopRendering()
        }
    }

    deinit {
        displayLink?.invalidate()
    }
}

// MARK: - Glass Thumb View
private class GlassThumbView: UIView {
    private var metalView: MTKView!
    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var pipelineState: MTLRenderPipelineState?
    private var displayLink: CADisplayLink?
    private var backgroundTexture: MTLTexture?
    private var isRenderingActive = false

    private var uniforms = ThumbUniforms()

    struct ThumbUniforms {
        var size: SIMD2<Float> = .zero
        var offset: SIMD2<Float> = .zero
        var backgroundSize: SIMD2<Float> = .zero
        var refractionStrength: Float = 3.0
        var chromaticAberration: Float = 15.0
        var pressedAmount: Float = 0.0
    }

    init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            super.init(frame: .zero)
            return
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!

        super.init(frame: .zero)

        self.backgroundColor = .clear
        self.isOpaque = false

        setupMetal()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    deinit {
        stopRendering()
    }

    private func setupMetal() {
        metalView = MTKView(frame: bounds, device: device)
        metalView.backgroundColor = .clear
        metalView.isOpaque = false
        metalView.framebufferOnly = false
        metalView.isPaused = true
        metalView.preferredFramesPerSecond = 60
        metalView.delegate = self
        addSubview(metalView)

        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct VertexOut {
            float4 position [[position]];
            float2 texCoord;
        };

        struct ThumbUniforms {
            float2 size;
            float2 offset;
            float2 backgroundSize;
            float refractionStrength;
            float chromaticAberration;
            float pressedAmount;
        };

        vertex VertexOut thumbVertex(uint vertexID [[vertex_id]]) {
            float2 positions[4] = {
                float2(-1.0, -1.0), float2( 1.0, -1.0),
                float2(-1.0,  1.0), float2( 1.0,  1.0)
            };
            float2 texCoords[4] = {
                float2(0.0, 1.0), float2(1.0, 1.0),
                float2(0.0, 0.0), float2(1.0, 0.0)
            };
            VertexOut out;
            out.position = float4(positions[vertexID], 0.0, 1.0);
            out.texCoord = texCoords[vertexID];
            return out;
        }

        // Signed distance for rounded rectangle (pill/capsule shape)
        float sdRoundedRect(float2 p, float2 b, float r) {
            float2 q = abs(p) - b + r;
            return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
        }

        fragment float4 thumbFragment(
            VertexOut in [[stage_in]],
            texture2d<float, access::sample> backgroundTexture [[texture(0)]],
            constant ThumbUniforms &uniforms [[buffer(0)]]
        ) {
            constexpr sampler texSampler(coord::normalized, address::clamp_to_edge, filter::linear);

            float2 uv = in.texCoord;
            float2 pixelPos = uv * uniforms.size;
            float2 center = uniforms.size * 0.5;
            float2 p = pixelPos - center;

            // Rounded rectangle / pill shape
            float2 halfSize = uniforms.size * 0.5 - 1.0;
            float radius = min(halfSize.x, halfSize.y);
            float sdf = sdRoundedRect(p, halfSize, radius);

            if (sdf > 1.5) discard_fragment();

            // === DISTANCE & EDGE CALCULATIONS ===
            float distFromEdge = -sdf;
            float edgeBand = radius * 0.5;
            float edgeFactor = 1.0 - smoothstep(0.0, edgeBand, distFromEdge);
            float centerFactor = smoothstep(0.0, radius, distFromEdge);

            // === 3D GLASS SURFACE NORMAL ===
            float2 normDir = length(p) > 0.001 ? normalize(p) : float2(0.0, 1.0);
            float depth = clamp(distFromEdge / radius, 0.0, 1.0);
            float zHeight = sqrt(max(0.0, 1.0 - pow(1.0 - depth, 2.0)));
            float3 N = normalize(float3(normDir * (1.0 - depth) * 0.7, zHeight));

            // === VIEW DIRECTION & FRESNEL ===
            float3 V = float3(0.0, 0.0, 1.0);
            float NdotV = max(dot(N, V), 0.0);
            float fresnel = pow(1.0 - NdotV, 3.0);

            // === INTERACTION BOOST ===
            float activeBoost = 1.0 + uniforms.pressedAmount * 0.4;

            // === REFRACTION - Strong light bending ===
            float ior = 1.5;
            float3 refractDir = refract(-V, N, 1.0 / ior);
            float refractionStrength = 25.0 * activeBoost * (0.6 + depth * 0.4);

            // === CHROMATIC ABERRATION - RGB split at edges ===
            float chromaticStrength = 12.0 * edgeFactor * activeBoost;

            float2 refractOffset = refractDir.xy * refractionStrength;
            float2 redOffset = refractOffset + normDir * chromaticStrength;
            float2 greenOffset = refractOffset;
            float2 blueOffset = refractOffset - normDir * chromaticStrength * 0.7;

            float2 uvR = (uniforms.offset + pixelPos + redOffset) / uniforms.backgroundSize;
            float2 uvG = (uniforms.offset + pixelPos + greenOffset) / uniforms.backgroundSize;
            float2 uvB = (uniforms.offset + pixelPos + blueOffset) / uniforms.backgroundSize;

            uvR = clamp(uvR, float2(0.002), float2(0.998));
            uvG = clamp(uvG, float2(0.002), float2(0.998));
            uvB = clamp(uvB, float2(0.002), float2(0.998));

            float3 col;
            col.r = backgroundTexture.sample(texSampler, uvR).r;
            col.g = backgroundTexture.sample(texSampler, uvG).g;
            col.b = backgroundTexture.sample(texSampler, uvB).b;

            // === CRYSTAL CLEAR GLASS ===
            col = mix(col, float3(1.0), 0.02 + fresnel * 0.05);

            // === EDGE COLOR - Subtle cyan chromatic ===
            float3 edgeColor = float3(0.9, 0.96, 1.0);
            col = mix(col, edgeColor, edgeFactor * edgeFactor * 0.2);

            // === SPECULAR HIGHLIGHTS - Sharp reflections ===
            float3 L1 = normalize(float3(-0.4, -0.6, 1.0));
            float3 H1 = normalize(L1 + V);
            float spec1 = pow(max(dot(N, H1), 0.0), 150.0);

            float3 L2 = normalize(float3(0.3, -0.5, 1.0));
            float3 H2 = normalize(L2 + V);
            float spec2 = pow(max(dot(N, H2), 0.0), 100.0);

            // Intensify highlights on press
            float specBoost = 1.0 + uniforms.pressedAmount * 0.5;
            col += spec1 * 0.7 * specBoost;
            col += spec2 * 0.3 * specBoost;

            // === FRESNEL RIM - Clear glass edge ===
            col += fresnel * 0.15 * float3(0.96, 0.98, 1.0);

            // === INNER GLOW ===
            col += centerFactor * 0.03;

            // === SUBTLE BOTTOM SHADOW ===
            float bottomShadow = smoothstep(-halfSize.y, halfSize.y * 0.5, p.y);
            col *= mix(0.95, 1.0, bottomShadow);

            // === ANTI-ALIASING ===
            float aa = 1.0 - smoothstep(-1.5, 1.5, sdf);

            return float4(col, aa * 0.85);
        }
        """

        guard let library = try? device.makeLibrary(source: shaderSource, options: nil),
              let vertexFunc = library.makeFunction(name: "thumbVertex"),
              let fragmentFunc = library.makeFunction(name: "thumbFragment") else {
            return
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunc
        descriptor.fragmentFunction = fragmentFunc
        descriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        descriptor.colorAttachments[0].isBlendingEnabled = true
        descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        descriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        pipelineState = try? device.makeRenderPipelineState(descriptor: descriptor)
    }

    func setPressed(_ pressed: Bool) {
        uniforms.pressedAmount = pressed ? 1.0 : 0.0
    }

    func startRendering() {
        guard !isRenderingActive else { return }
        isRenderingActive = true
        metalView.isPaused = false

        let link = CADisplayLink(target: self, selector: #selector(updateFrame))
        link.preferredFramesPerSecond = 20
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    func stopRendering() {
        isRenderingActive = false
        metalView.isPaused = true
        displayLink?.invalidate()
        displayLink = nil
    }

    @objc private func updateFrame() {
        captureBackground()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        metalView.frame = bounds
    }

    private func captureBackground() {
        guard let window = self.window else { return }
        guard bounds.width > 0, bounds.height > 0 else { return }

        let frameInWindow = convert(bounds, to: window)
        let padding: CGFloat = 10.0
        let captureRect = frameInWindow.insetBy(dx: -padding, dy: -padding)

        guard captureRect.width > 0, captureRect.height > 0 else { return }

        let scale = UIScreen.main.scale
        UIGraphicsBeginImageContextWithOptions(captureRect.size, false, scale)
        defer { UIGraphicsEndImageContext() }

        guard let ctx = UIGraphicsGetCurrentContext() else { return }

        ctx.translateBy(x: -captureRect.origin.x, y: -captureRect.origin.y)

        let savedAlpha = alpha
        alpha = 0
        window.layer.render(in: ctx)
        alpha = savedAlpha

        guard let snapshot = UIGraphicsGetImageFromCurrentImageContext(),
              let cgImage = snapshot.cgImage else { return }

        let textureLoader = MTKTextureLoader(device: device)
        backgroundTexture = try? textureLoader.newTexture(cgImage: cgImage, options: [.SRGB: false])

        let offset = CGPoint(
            x: frameInWindow.origin.x - captureRect.origin.x,
            y: frameInWindow.origin.y - captureRect.origin.y
        )

        uniforms.size = SIMD2<Float>(Float(bounds.width * scale), Float(bounds.height * scale))
        uniforms.offset = SIMD2<Float>(Float(offset.x * scale), Float(offset.y * scale))
        uniforms.backgroundSize = SIMD2<Float>(Float(captureRect.width * scale), Float(captureRect.height * scale))

        metalView.setNeedsDisplay()
    }
}

extension GlassThumbView: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let pipelineState = pipelineState,
              let drawable = view.currentDrawable,
              let descriptor = view.currentRenderPassDescriptor,
              let backgroundTexture = backgroundTexture else { return }

        descriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        descriptor.colorAttachments[0].loadAction = .clear

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }

        encoder.setRenderPipelineState(pipelineState)
        encoder.setFragmentTexture(backgroundTexture, index: 0)
        var uniformsCopy = uniforms
        encoder.setFragmentBytes(&uniformsCopy, length: MemoryLayout<ThumbUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

// MARK: - Glass Track View
private class GlassTrackView: UIView {
    private var metalView: MTKView!
    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var pipelineState: MTLRenderPipelineState?
    private var displayLink: CADisplayLink?
    private var backgroundTexture: MTLTexture?
    private var isRenderingActive = false

    private var uniforms = TrackUniforms()

    struct TrackUniforms {
        var size: SIMD2<Float> = .zero
        var offset: SIMD2<Float> = .zero
        var backgroundSize: SIMD2<Float> = .zero
        var refractionStrength: Float = 2.0
        var chromaticAberration: Float = 12.0
    }

    init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            super.init(frame: .zero)
            return
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!

        super.init(frame: .zero)

        self.backgroundColor = .clear
        self.isOpaque = false

        setupMetal()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    deinit {
        stopRendering()
    }

    private func setupMetal() {
        metalView = MTKView(frame: bounds, device: device)
        metalView.backgroundColor = .clear
        metalView.isOpaque = false
        metalView.framebufferOnly = false
        metalView.isPaused = true
        metalView.preferredFramesPerSecond = 60
        metalView.delegate = self
        addSubview(metalView)

        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct VertexOut {
            float4 position [[position]];
            float2 texCoord;
        };

        struct TrackUniforms {
            float2 size;
            float2 offset;
            float2 backgroundSize;
            float refractionStrength;
            float chromaticAberration;
        };

        vertex VertexOut trackVertex(uint vertexID [[vertex_id]]) {
            float2 positions[4] = {
                float2(-1.0, -1.0), float2( 1.0, -1.0),
                float2(-1.0,  1.0), float2( 1.0,  1.0)
            };
            float2 texCoords[4] = {
                float2(0.0, 1.0), float2(1.0, 1.0),
                float2(0.0, 0.0), float2(1.0, 0.0)
            };
            VertexOut out;
            out.position = float4(positions[vertexID], 0.0, 1.0);
            out.texCoord = texCoords[vertexID];
            return out;
        }

        float sdRoundedBox(float2 p, float2 size, float radius) {
            float2 d = abs(p) - size + radius;
            return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - radius;
        }

        fragment float4 trackFragment(
            VertexOut in [[stage_in]],
            texture2d<float, access::sample> backgroundTexture [[texture(0)]],
            constant TrackUniforms &uniforms [[buffer(0)]]
        ) {
            constexpr sampler texSampler(coord::normalized, address::clamp_to_edge, filter::linear);

            float2 uv = in.texCoord;
            float2 pixelPos = uv * uniforms.size;
            float2 center = uniforms.size * 0.5;
            float2 p = pixelPos - center;

            float cornerRadius = uniforms.size.y * 0.5;
            float2 boxSize = uniforms.size * 0.5;
            float sdf = sdRoundedBox(p, boxSize, cornerRadius);

            float borderWidth = 2.5;
            float outerEdge = sdf;
            float innerEdge = sdf + borderWidth;

            if (outerEdge > 1.0) discard_fragment();
            if (innerEdge < -1.0) discard_fragment();

            // Border position (0 = outer, 1 = inner)
            float borderPos = clamp(-sdf / borderWidth, 0.0, 1.0);

            // Normal pointing outward from track center
            float2 normDir = length(p) > 0.001 ? normalize(p) : float2(1.0, 0.0);

            // 3D normal for the border tube
            float tubeAngle = borderPos * 3.14159;
            float3 N = normalize(float3(normDir * sin(tubeAngle), cos(tubeAngle)));

            float3 V = float3(0.0, 0.0, 1.0);
            float fresnel = pow(1.0 - max(dot(N, V), 0.0), 2.5);

            // === REFRACTION ===
            float3 refractDir = refract(-V, N, 1.0 / 1.4);
            float refractionStrength = 15.0 * sin(tubeAngle);

            // === CHROMATIC ABERRATION ===
            float chromeStr = 8.0 * (1.0 - borderPos);

            float2 refractOffset = refractDir.xy * refractionStrength;
            float2 redOffset = refractOffset + normDir * chromeStr;
            float2 greenOffset = refractOffset;
            float2 blueOffset = refractOffset - normDir * chromeStr * 0.7;

            float2 uvR = (uniforms.offset + pixelPos + redOffset) / uniforms.backgroundSize;
            float2 uvG = (uniforms.offset + pixelPos + greenOffset) / uniforms.backgroundSize;
            float2 uvB = (uniforms.offset + pixelPos + blueOffset) / uniforms.backgroundSize;

            uvR = clamp(uvR, float2(0.002), float2(0.998));
            uvG = clamp(uvG, float2(0.002), float2(0.998));
            uvB = clamp(uvB, float2(0.002), float2(0.998));

            float3 col;
            col.r = backgroundTexture.sample(texSampler, uvR).r;
            col.g = backgroundTexture.sample(texSampler, uvG).g;
            col.b = backgroundTexture.sample(texSampler, uvB).b;

            // === CLEAR GLASS ===
            col = mix(col, float3(1.0), 0.02 + fresnel * 0.04);

            // === EDGE TINT ===
            col = mix(col, float3(0.92, 0.96, 1.0), (1.0 - borderPos) * 0.15);

            // === SPECULAR ===
            float3 L = normalize(float3(-0.4, -0.6, 1.0));
            float3 H = normalize(L + V);
            float spec = pow(max(dot(N, H), 0.0), 100.0);
            col += spec * 0.5;

            // === FRESNEL RIM ===
            col += fresnel * 0.1;

            // === ANTI-ALIASING ===
            float aaOuter = smoothstep(1.0, -1.0, outerEdge);
            float aaInner = 1.0 - smoothstep(-1.0, 1.0, innerEdge);
            float aa = aaOuter * aaInner;

            return float4(col, aa * 0.8);
        }
        """

        guard let library = try? device.makeLibrary(source: shaderSource, options: nil),
              let vertexFunc = library.makeFunction(name: "trackVertex"),
              let fragmentFunc = library.makeFunction(name: "trackFragment") else {
            return
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunc
        descriptor.fragmentFunction = fragmentFunc
        descriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        descriptor.colorAttachments[0].isBlendingEnabled = true
        descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        descriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        pipelineState = try? device.makeRenderPipelineState(descriptor: descriptor)
    }

    func startRendering() {
        guard !isRenderingActive else { return }
        isRenderingActive = true
        metalView.isPaused = false

        let link = CADisplayLink(target: self, selector: #selector(updateFrame))
        link.preferredFramesPerSecond = 20
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    func stopRendering() {
        isRenderingActive = false
        metalView.isPaused = true
        displayLink?.invalidate()
        displayLink = nil
    }

    @objc private func updateFrame() {
        captureBackground()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        metalView.frame = bounds
    }

    private func captureBackground() {
        guard let window = self.window else { return }
        guard bounds.width > 0, bounds.height > 0 else { return }

        let frameInWindow = convert(bounds, to: window)
        let padding: CGFloat = 10.0
        let captureRect = frameInWindow.insetBy(dx: -padding, dy: -padding)

        guard captureRect.width > 0, captureRect.height > 0 else { return }

        let scale = UIScreen.main.scale
        UIGraphicsBeginImageContextWithOptions(captureRect.size, false, scale)
        defer { UIGraphicsEndImageContext() }

        guard let ctx = UIGraphicsGetCurrentContext() else { return }

        ctx.translateBy(x: -captureRect.origin.x, y: -captureRect.origin.y)

        let savedAlpha = alpha
        alpha = 0
        window.layer.render(in: ctx)
        alpha = savedAlpha

        guard let snapshot = UIGraphicsGetImageFromCurrentImageContext(),
              let cgImage = snapshot.cgImage else { return }

        let textureLoader = MTKTextureLoader(device: device)
        backgroundTexture = try? textureLoader.newTexture(cgImage: cgImage, options: [.SRGB: false])

        let offset = CGPoint(
            x: frameInWindow.origin.x - captureRect.origin.x,
            y: frameInWindow.origin.y - captureRect.origin.y
        )

        uniforms.size = SIMD2<Float>(Float(bounds.width * scale), Float(bounds.height * scale))
        uniforms.offset = SIMD2<Float>(Float(offset.x * scale), Float(offset.y * scale))
        uniforms.backgroundSize = SIMD2<Float>(Float(captureRect.width * scale), Float(captureRect.height * scale))

        metalView.setNeedsDisplay()
    }
}

extension GlassTrackView: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let pipelineState = pipelineState,
              let drawable = view.currentDrawable,
              let descriptor = view.currentRenderPassDescriptor,
              let backgroundTexture = backgroundTexture else { return }

        descriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        descriptor.colorAttachments[0].loadAction = .clear

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }

        encoder.setRenderPipelineState(pipelineState)
        encoder.setFragmentTexture(backgroundTexture, index: 0)
        var uniformsCopy = uniforms
        encoder.setFragmentBytes(&uniformsCopy, length: MemoryLayout<TrackUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
