import Foundation
import UIKit
import Display
import AsyncDisplayKit
import TelegramPresentationData
import LegacyComponents
import ComponentFlow
import MetalKit

public final class SliderComponent: Component {
    public final class Discrete: Equatable {
        public let valueCount: Int
        public let value: Int
        public let minValue: Int?
        public let markPositions: Bool
        public let valueUpdated: (Int) -> Void
        
        public init(valueCount: Int, value: Int, minValue: Int? = nil, markPositions: Bool, valueUpdated: @escaping (Int) -> Void) {
            self.valueCount = valueCount
            self.value = value
            self.minValue = minValue
            self.markPositions = markPositions
            self.valueUpdated = valueUpdated
        }
        
        public static func ==(lhs: Discrete, rhs: Discrete) -> Bool {
            if lhs.valueCount != rhs.valueCount {
                return false
            }
            if lhs.value != rhs.value {
                return false
            }
            if lhs.minValue != rhs.minValue {
                return false
            }
            if lhs.markPositions != rhs.markPositions {
                return false
            }
            return true
        }
    }
    
    public final class Continuous: Equatable {
        public let value: CGFloat
        public let minValue: CGFloat?
        public let valueUpdated: (CGFloat) -> Void
        
        public init(value: CGFloat, minValue: CGFloat? = nil, valueUpdated: @escaping (CGFloat) -> Void) {
            self.value = value
            self.minValue = minValue
            self.valueUpdated = valueUpdated
        }
        
        public static func ==(lhs: Continuous, rhs: Continuous) -> Bool {
            if lhs.value != rhs.value {
                return false
            }
            if lhs.minValue != rhs.minValue {
                return false
            }
            return true
        }
    }
    
    public enum Content: Equatable {
        case discrete(Discrete)
        case continuous(Continuous)
    }
    
    public let content: Content
    public let useNative: Bool
    public let useLiquidGlass: Bool
    public let trackBackgroundColor: UIColor
    public let trackForegroundColor: UIColor
    public let minTrackForegroundColor: UIColor?
    public let knobSize: CGFloat?
    public let knobColor: UIColor?
    public let isTrackingUpdated: ((Bool) -> Void)?

    public init(
        content: Content,
        useNative: Bool = false,
        useLiquidGlass: Bool = false,
        trackBackgroundColor: UIColor,
        trackForegroundColor: UIColor,
        minTrackForegroundColor: UIColor? = nil,
        knobSize: CGFloat? = nil,
        knobColor: UIColor? = nil,
        isTrackingUpdated: ((Bool) -> Void)? = nil
    ) {
        self.content = content
        self.useNative = useNative
        self.useLiquidGlass = useLiquidGlass
        self.trackBackgroundColor = trackBackgroundColor
        self.trackForegroundColor = trackForegroundColor
        self.minTrackForegroundColor = minTrackForegroundColor
        self.knobSize = knobSize
        self.knobColor = knobColor
        self.isTrackingUpdated = isTrackingUpdated
    }
    
    public static func ==(lhs: SliderComponent, rhs: SliderComponent) -> Bool {
        if lhs.content != rhs.content {
            return false
        }
        if lhs.useLiquidGlass != rhs.useLiquidGlass {
            return false
        }
        if lhs.trackBackgroundColor != rhs.trackBackgroundColor {
            return false
        }
        if lhs.trackForegroundColor != rhs.trackForegroundColor {
            return false
        }
        if lhs.minTrackForegroundColor != rhs.minTrackForegroundColor {
            return false
        }
        if lhs.knobSize != rhs.knobSize {
            return false
        }
        if lhs.knobColor != rhs.knobColor {
            return false
        }
        return true
    }
    
    final class SliderView: UISlider {

    }

    private final class LiquidGlassKnobView: UIView, MTKViewDelegate {
        private var solidKnob: UIView!

        private var metalView: MTKView!
        private var device: MTLDevice!
        private var commandQueue: MTLCommandQueue!
        private var pipelineState: MTLRenderPipelineState?
        private var displayLink: CADisplayLink?
        private var backgroundTexture: MTLTexture?

        private var isPressed = false
        private var isGlassVisible = false

        private var uniforms = KnobUniforms()

        struct KnobUniforms {
            var size: SIMD2<Float> = .zero
            var offset: SIMD2<Float> = .zero
            var backgroundSize: SIMD2<Float> = .zero
            var cornerRadius: Float = 12.0
            var edgeSoftness: Float = 1.5
        }

        init() {
            super.init(frame: .zero)

            self.backgroundColor = .clear
            self.isOpaque = false

            solidKnob = UIView()
            solidKnob.backgroundColor = .white
            solidKnob.layer.shadowColor = UIColor.black.cgColor
            solidKnob.layer.shadowOffset = CGSize(width: 0, height: 4)
            solidKnob.layer.shadowRadius = 8
            solidKnob.layer.shadowOpacity = 0.12
            addSubview(solidKnob)

            guard let device = MTLCreateSystemDefaultDevice() else {
                return
            }
            self.device = device
            self.commandQueue = device.makeCommandQueue()!

            setupMetal()
        }

        required init?(coder: NSCoder) {
            fatalError("init(coder:) has not been implemented")
        }

        deinit {
            stopDisplayLink()
        }

        private func setupMetal() {
            metalView = MTKView(frame: bounds, device: device)
            metalView.backgroundColor = .clear
            metalView.isOpaque = false
            metalView.framebufferOnly = false
            metalView.isPaused = true
            metalView.preferredFramesPerSecond = 60
            metalView.delegate = self
            metalView.alpha = 0
            addSubview(metalView)

            let shaderSource = """
            #include <metal_stdlib>
            using namespace metal;

            struct VertexOut {
                float4 position [[position]];
                float2 texCoord;
            };

            struct KnobUniforms {
                float2 size;
                float2 offset;
                float2 backgroundSize;
                float cornerRadius;
                float edgeSoftness;
            };

            vertex VertexOut knobVertex(uint vertexID [[vertex_id]]) {
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

            // Signed distance function for rounded rectangle (pill shape)
            float sdRoundedRect(float2 p, float2 b, float r) {
                float2 q = abs(p) - b + r;
                return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
            }

            fragment float4 knobFragment(
                VertexOut in [[stage_in]],
                texture2d<float, access::sample> backgroundTexture [[texture(0)]],
                constant KnobUniforms &uniforms [[buffer(0)]]
            ) {
                constexpr sampler texSampler(coord::normalized, address::clamp_to_edge, filter::linear);

                float2 uv = in.texCoord;
                float2 pixelPos = uv * uniforms.size;
                float2 center = uniforms.size * 0.5;
                float2 p = pixelPos - center;

                // Horizontal pill shape
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
                // Treat as a convex lens/bubble shape
                float2 normDir = length(p) > 0.001 ? normalize(p) : float2(0.0, 1.0);
                float depth = clamp(distFromEdge / radius, 0.0, 1.0);
                float zHeight = sqrt(max(0.0, 1.0 - pow(1.0 - depth, 2.0)));
                float3 N = normalize(float3(normDir * (1.0 - depth) * 0.7, zHeight));

                // === VIEW DIRECTION & FRESNEL ===
                float3 V = float3(0.0, 0.0, 1.0);
                float NdotV = max(dot(N, V), 0.0);
                float fresnel = pow(1.0 - NdotV, 3.0);

                // === REFRACTION - Strong light bending through glass ===
                float ior = 1.5;  // Glass index of refraction
                float3 refractDir = refract(-V, N, 1.0 / ior);
                float refractionStrength = 25.0 * (0.6 + depth * 0.4);

                // === CHROMATIC ABERRATION - RGB channel separation at edges ===
                float chromaticStrength = 12.0 * edgeFactor;

                // Red shifts outward, blue shifts inward
                float2 refractOffset = refractDir.xy * refractionStrength;
                float2 redOffset = refractOffset + normDir * chromaticStrength;
                float2 greenOffset = refractOffset;
                float2 blueOffset = refractOffset - normDir * chromaticStrength * 0.7;

                // Sample background with chromatic aberration
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

                // === CRYSTAL CLEAR GLASS - Minimal tint ===
                col = mix(col, float3(1.0), 0.02 + fresnel * 0.05);

                // === EDGE COLOR - Subtle cyan chromatic edge ===
                float3 edgeColor = float3(0.9, 0.96, 1.0);
                col = mix(col, edgeColor, edgeFactor * edgeFactor * 0.2);

                // === SPECULAR HIGHLIGHTS - Sharp glass reflections ===
                float3 L1 = normalize(float3(-0.4, -0.6, 1.0));
                float3 H1 = normalize(L1 + V);
                float spec1 = pow(max(dot(N, H1), 0.0), 150.0);

                float3 L2 = normalize(float3(0.3, -0.5, 1.0));
                float3 H2 = normalize(L2 + V);
                float spec2 = pow(max(dot(N, H2), 0.0), 100.0);

                col += spec1 * 0.7;
                col += spec2 * 0.3;

                // === FRESNEL RIM - Clear glass edge ===
                col += fresnel * 0.12 * float3(0.96, 0.98, 1.0);

                // === SUBTLE BOTTOM SHADOW ===
                float bottomShadow = smoothstep(-halfSize.y, halfSize.y * 0.5, p.y);
                col *= mix(0.95, 1.0, bottomShadow);

                // === ANTI-ALIASING ===
                float aa = 1.0 - smoothstep(-1.5, 1.5, sdf);

                // Very clear glass
                return float4(col, aa * 0.82);
            }
            """

            guard let library = try? device.makeLibrary(source: shaderSource, options: nil),
                  let vertexFunc = library.makeFunction(name: "knobVertex"),
                  let fragmentFunc = library.makeFunction(name: "knobFragment") else {
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

        // MARK: - Morphing Control

        /// Call this when user starts interacting with slider
        func setPressed(_ pressed: Bool) {
            guard isPressed != pressed else { return }
            isPressed = pressed

            if pressed {
                morphToGlass()
            } else {
                morphToSolid()
            }
        }

        private func morphToGlass() {
            guard !isGlassVisible else { return }
            isGlassVisible = true

            startDisplayLink()
            metalView.isPaused = false

            UIView.animate(withDuration: 0.2, delay: 0, options: [.curveEaseOut], animations: {
                self.solidKnob.alpha = 0
                self.metalView.alpha = 1
            })
        }

        private func morphToSolid() {
            guard isGlassVisible else { return }
            isGlassVisible = false

            UIView.animate(withDuration: 0.25, delay: 0, options: [.curveEaseIn], animations: {
                self.solidKnob.alpha = 1
                self.metalView.alpha = 0
            }, completion: { _ in
                if !self.isGlassVisible {
                    self.metalView.isPaused = true
                    self.stopDisplayLink()
                }
            })
        }

        override func didMoveToWindow() {
            super.didMoveToWindow()
            if window != nil && isGlassVisible {
                startDisplayLink()
            } else if window == nil {
                stopDisplayLink()
            }
        }

        private func startDisplayLink() {
            guard displayLink == nil else { return }
            let link = CADisplayLink(target: self, selector: #selector(updateFrame))
            link.preferredFramesPerSecond = 20  // Background capture at 20 FPS
            link.add(to: .main, forMode: .common)
            displayLink = link
        }

        private func stopDisplayLink() {
            displayLink?.invalidate()
            displayLink = nil
        }

        @objc private func updateFrame() {
            if isGlassVisible {
                captureAndRender()
            }
        }

        override func layoutSubviews() {
            super.layoutSubviews()

            solidKnob.frame = bounds
            solidKnob.layer.cornerRadius = min(bounds.width, bounds.height) / 2

            metalView?.frame = bounds
        }

        private func captureAndRender() {
            guard let window = self.window else { return }
            guard bounds.width > 0, bounds.height > 0 else { return }

            let frameInWindow = convert(bounds, to: window)
            let padding: CGFloat = 30.0  // Larger padding for chromatic aberration
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
            uniforms.cornerRadius = Float(min(bounds.width, bounds.height) / 2 * scale)
            uniforms.edgeSoftness = Float(1.5 * scale)

            metalView.setNeedsDisplay()
        }

        // MARK: - MTKViewDelegate
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
            encoder.setFragmentBytes(&uniformsCopy, length: MemoryLayout<KnobUniforms>.stride, index: 0)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            encoder.endEncoding()

            commandBuffer.present(drawable)
            commandBuffer.commit()
        }
    }

    public final class View: UIView {
        private var nativeSliderView: SliderView?
        private var sliderView: TGPhotoEditorSliderView?
        private var liquidGlassKnob: LiquidGlassKnobView?

        private var component: SliderComponent?
        private weak var state: EmptyComponentState?
        
        public var hitTestTarget: UIView? {
            return self.sliderView
        }
        
        override public init(frame: CGRect) {
            super.init(frame: frame)
        }
        
        required public init?(coder: NSCoder) {
            fatalError("init(coder:) has not been implemented")
        }
                
        public func cancelGestures() {
            if let sliderView = self.sliderView, let gestureRecognizers = sliderView.gestureRecognizers {
                for gestureRecognizer in gestureRecognizers {
                    if gestureRecognizer.isEnabled {
                        gestureRecognizer.isEnabled = false
                        gestureRecognizer.isEnabled = true
                    }
                }
            }
        }
        
        func update(component: SliderComponent, availableSize: CGSize, state: EmptyComponentState, environment: Environment<Empty>, transition: ComponentTransition) -> CGSize {
            self.component = component
            self.state = state
            
            let size = CGSize(width: availableSize.width, height: 44.0)
            
            if #available(iOS 26.0, *), component.useNative {
                let sliderView: SliderView
                if let current = self.nativeSliderView {
                    sliderView = current
                } else {
                    sliderView = SliderView()
                    sliderView.disablesInteractiveTransitionGestureRecognizer = true
                    sliderView.addTarget(self, action: #selector(self.sliderValueChanged), for: .valueChanged)
                    sliderView.layer.allowsGroupOpacity = true
                    
                    self.addSubview(sliderView)
                    self.nativeSliderView = sliderView
                    
                    switch component.content {
                    case let .continuous(continuous):
                        sliderView.minimumValue = Float(continuous.minValue ?? 0.0)
                        sliderView.maximumValue = 1.0
                    case let .discrete(discrete):
                        sliderView.minimumValue = 0.0
                        sliderView.maximumValue = Float(discrete.valueCount - 1)
                        sliderView.trackConfiguration = .init(numberOfTicks: discrete.valueCount)
                    }
                }
                switch component.content {
                case let .continuous(continuous):
                    sliderView.value = Float(continuous.value)
                case let .discrete(discrete):
                    sliderView.value = Float(discrete.value)
                }
                sliderView.minimumTrackTintColor = component.trackForegroundColor
                sliderView.maximumTrackTintColor = component.trackBackgroundColor
                
                transition.setFrame(view: sliderView, frame: CGRect(origin: CGPoint(x: 0.0, y: 0.0), size: CGSize(width: availableSize.width, height: 44.0)))
            } else {
                var internalIsTrackingUpdated: ((Bool) -> Void)?
                if let isTrackingUpdated = component.isTrackingUpdated {
                    internalIsTrackingUpdated = { [weak self] isTracking in
                        if let self {
                            if !"".isEmpty {
                                if isTracking {
                                    self.sliderView?.bordered = true
                                } else {
                                    DispatchQueue.main.asyncAfter(deadline: DispatchTime.now() + 0.1, execute: { [weak self] in
                                        self?.sliderView?.bordered = false
                                    })
                                }
                            }
                        }
                        isTrackingUpdated(isTracking)
                    }
                }
                
                let sliderView: TGPhotoEditorSliderView
                if let current = self.sliderView {
                    sliderView = current
                } else {
                    sliderView = TGPhotoEditorSliderView()
                    sliderView.enablePanHandling = true
                    if let knobSize = component.knobSize {
                        sliderView.lineSize = knobSize + 4.0
                    } else {
                        sliderView.lineSize = 4.0
                    }
                    sliderView.trackCornerRadius = sliderView.lineSize * 0.5
                    sliderView.dotSize = 5.0
                    sliderView.minimumValue = 0.0
                    sliderView.startValue = 0.0
                    sliderView.disablesInteractiveTransitionGestureRecognizer = true
                    
                    switch component.content {
                    case let .discrete(discrete):
                        sliderView.maximumValue = CGFloat(discrete.valueCount - 1)
                        sliderView.positionsCount = discrete.valueCount
                        sliderView.useLinesForPositions = true
                        sliderView.markPositions = discrete.markPositions
                    case .continuous:
                        sliderView.maximumValue = 1.0
                    }
                    
                    sliderView.backgroundColor = nil
                    sliderView.isOpaque = false
                    sliderView.backColor = component.trackBackgroundColor
                    sliderView.startColor = component.trackBackgroundColor
                    sliderView.trackColor = component.trackForegroundColor

                    if component.useLiquidGlass {
                        sliderView.knobImage = generateImage(CGSize(width: 40.0, height: 40.0), rotatedContext: { size, context in
                            context.clear(CGRect(origin: CGPoint(), size: size))
                        })
                    } else if let knobSize = component.knobSize {
                        sliderView.knobImage = generateImage(CGSize(width: 40.0, height: 40.0), rotatedContext: { size, context in
                            context.clear(CGRect(origin: CGPoint(), size: size))
                            context.setShadow(offset: CGSize(width: 0.0, height: -3.0), blur: 12.0, color: UIColor(white: 0.0, alpha: 0.25).cgColor)
                            if let knobColor = component.knobColor {
                                context.setFillColor(knobColor.cgColor)
                            } else {
                                context.setFillColor(UIColor.white.cgColor)
                            }
                            context.fillEllipse(in: CGRect(origin: CGPoint(x: floor((size.width - knobSize) * 0.5), y: floor((size.width - knobSize) * 0.5)), size: CGSize(width: knobSize, height: knobSize)))
                        })
                    } else {
                        sliderView.knobImage = generateImage(CGSize(width: 40.0, height: 40.0), rotatedContext: { size, context in
                            context.clear(CGRect(origin: CGPoint(), size: size))
                            context.setShadow(offset: CGSize(width: 0.0, height: -3.0), blur: 12.0, color: UIColor(white: 0.0, alpha: 0.25).cgColor)
                            context.setFillColor(UIColor.white.cgColor)
                            context.fillEllipse(in: CGRect(origin: CGPoint(x: 6.0, y: 6.0), size: CGSize(width: 28.0, height: 28.0)))
                        })
                    }
                    
                    sliderView.frame = CGRect(origin: CGPoint(x: 0.0, y: 0.0), size: size)
                    sliderView.hitTestEdgeInsets = UIEdgeInsets(top: -sliderView.frame.minX, left: 0.0, bottom: 0.0, right: -sliderView.frame.minX)
                    
                    
                    sliderView.disablesInteractiveTransitionGestureRecognizer = true
                    sliderView.addTarget(self, action: #selector(self.sliderValueChanged), for: .valueChanged)
                    sliderView.layer.allowsGroupOpacity = true
                    self.sliderView = sliderView
                    self.addSubview(sliderView)
                }
                sliderView.lowerBoundTrackColor = component.minTrackForegroundColor
                switch component.content {
                case let .discrete(discrete):
                    sliderView.value = CGFloat(discrete.value)
                    if let minValue = discrete.minValue {
                        sliderView.lowerBoundValue = CGFloat(minValue)
                    } else {
                        sliderView.lowerBoundValue = 0.0
                    }
                case let .continuous(continuous):
                    sliderView.value = continuous.value
                    if let minValue = continuous.minValue {
                        sliderView.lowerBoundValue = minValue
                    } else {
                        sliderView.lowerBoundValue = 0.0
                    }
                }
                sliderView.interactionBegan = { [weak self] in
                    self?.liquidGlassKnob?.setPressed(true)
                    internalIsTrackingUpdated?(true)
                }
                sliderView.interactionEnded = { [weak self] in
                    self?.liquidGlassKnob?.setPressed(false)
                    internalIsTrackingUpdated?(false)
                }

                transition.setFrame(view: sliderView, frame: CGRect(origin: CGPoint(x: 0.0, y: 0.0), size: CGSize(width: availableSize.width, height: 44.0)))
                sliderView.hitTestEdgeInsets = UIEdgeInsets(top: 0.0, left: 0.0, bottom: 0.0, right: 0.0)

                if component.useLiquidGlass {
                    let glassKnob: LiquidGlassKnobView
                    if let current = self.liquidGlassKnob {
                        glassKnob = current
                    } else {
                        glassKnob = LiquidGlassKnobView()
                        glassKnob.isUserInteractionEnabled = false
                        self.liquidGlassKnob = glassKnob
                        self.addSubview(glassKnob)
                    }

                    let knobWidth: CGFloat = 48.0
                    let knobHeight: CGFloat = 32.0
                    let trackWidth = availableSize.width - knobWidth
                    let trackX = knobWidth / 2

                    let normalizedValue: CGFloat
                    switch component.content {
                    case let .discrete(discrete):
                        normalizedValue = discrete.valueCount > 1 ? CGFloat(discrete.value) / CGFloat(discrete.valueCount - 1) : 0.0
                    case let .continuous(continuous):
                        normalizedValue = continuous.value
                    }

                    let knobCenterX = trackX + (trackWidth * normalizedValue)
                    let knobFrame = CGRect(
                        x: knobCenterX - knobWidth / 2,
                        y: (size.height - knobHeight) / 2,
                        width: knobWidth,
                        height: knobHeight
                    )

                    transition.setFrame(view: glassKnob, frame: knobFrame)
                } else {
                    if let glassKnob = self.liquidGlassKnob {
                        self.liquidGlassKnob = nil
                        glassKnob.removeFromSuperview()
                    }
                }
            }

            return size
        }
        
        @objc private func sliderValueChanged() {
            guard let component = self.component else {
                return
            }
            let floatValue: CGFloat
            if let sliderView = self.sliderView {
                floatValue = sliderView.value
            } else if let nativeSliderView = self.nativeSliderView {
                floatValue = CGFloat(nativeSliderView.value)
            } else {
                return
            }

            if component.useLiquidGlass, let glassKnob = self.liquidGlassKnob {
                let knobWidth: CGFloat = 48.0
                let knobHeight: CGFloat = 32.0
                let trackWidth = bounds.width - knobWidth
                let trackX = knobWidth / 2

                let normalizedValue: CGFloat
                switch component.content {
                case let .discrete(discrete):
                    normalizedValue = discrete.valueCount > 1 ? floatValue / CGFloat(discrete.valueCount - 1) : 0.0
                case .continuous:
                    normalizedValue = floatValue
                }

                let knobCenterX = trackX + (trackWidth * normalizedValue)
                let knobFrame = CGRect(
                    x: knobCenterX - knobWidth / 2,
                    y: (bounds.height - knobHeight) / 2,
                    width: knobWidth,
                    height: knobHeight
                )

                UIView.animate(withDuration: 0.1, delay: 0, options: [.curveLinear, .beginFromCurrentState], animations: {
                    glassKnob.frame = knobFrame
                })
            }

            switch component.content {
            case let .discrete(discrete):
                discrete.valueUpdated(Int(floatValue))
            case let .continuous(continuous):
                continuous.valueUpdated(floatValue)
            }
        }
    }

    public func makeView() -> View {
        return View(frame: CGRect())
    }
    
    public func update(view: View, availableSize: CGSize, state: EmptyComponentState, environment: Environment<Empty>, transition: ComponentTransition) -> CGSize {
        return view.update(component: self, availableSize: availableSize, state: state, environment: environment, transition: transition)
    }
}
