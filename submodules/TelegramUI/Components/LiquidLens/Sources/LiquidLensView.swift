import Foundation
import UIKit
import Display
import ComponentFlow
import GlassBackgroundComponent
import MetalKit

// MARK: - Lens Metal Implementation

private struct LensShaderSource {
    static let code = """
        #include <metal_stdlib>
        using namespace metal;

        struct VertexOut {
            float4 position [[position]];
            float2 texCoord;
        };

        struct LensUniforms {
            float2 size;
            float2 offset;
            float2 backgroundSize;
            float cornerRadius;
            float refractionStrength;
            float magnificationStrength;
            float liftAmount;
            float isDarkMode;
        };

        vertex VertexOut lensVertex(uint vertexID [[vertex_id]]) {
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

        float sdRoundedBox(float2 p, float2 b, float r) {
            float2 q = abs(p) - b + r;
            return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
        }

        fragment float4 lensFragment(
            VertexOut in [[stage_in]],
            texture2d<float, access::sample> backgroundTexture [[texture(0)]],
            constant LensUniforms &uniforms [[buffer(0)]]
        ) {
            constexpr sampler texSampler(coord::normalized, address::clamp_to_edge, filter::linear);

            float2 uv = in.texCoord;
            float2 pixelPos = uv * uniforms.size;
            float2 center = uniforms.size * 0.5;
            float2 fromCenter = pixelPos - center;

            float halfWidth = uniforms.size.x * 0.5;
            float halfHeight = uniforms.size.y * 0.5;
            float sdf = sdRoundedBox(fromCenter, float2(halfWidth, halfHeight), uniforms.cornerRadius);

            if (sdf > 0.0) {
                discard_fragment();
            }

            // Magnification
            float distFromCenter = length(fromCenter);
            float maxDist = min(halfWidth, halfHeight);
            float centerFactor = 1.0 - smoothstep(0.0, maxDist, distFromCenter);
            centerFactor = centerFactor * centerFactor;
            float magnification = 1.0 + (centerFactor * uniforms.magnificationStrength * uniforms.liftAmount);
            float2 magnifiedFromCenter = fromCenter / magnification;
            float2 magnifiedPixelPos = center + magnifiedFromCenter;

            // Refraction
            float distFromEdge = -sdf;
            float edgeBand = min(uniforms.size.x, uniforms.size.y) * 0.3;
            float edgeFactor = 1.0 - smoothstep(0.0, edgeBand, distFromEdge);
            edgeFactor = edgeFactor * edgeFactor * edgeFactor * 2.0;

            float2 nearestCenterPoint;
            float insetX = min(uniforms.cornerRadius, halfWidth);
            float insetY = min(uniforms.cornerRadius, halfHeight);

            if (uniforms.size.x >= uniforms.size.y) {
                float clampedX = clamp(magnifiedPixelPos.x, insetX, uniforms.size.x - insetX);
                nearestCenterPoint = float2(clampedX, center.y);
            } else {
                float clampedY = clamp(magnifiedPixelPos.y, insetY, uniforms.size.y - insetY);
                nearestCenterPoint = float2(center.x, clampedY);
            }

            float2 toCenter = nearestCenterPoint - magnifiedPixelPos;
            float distToCenter = length(toCenter);
            if (distToCenter > 0.001) {
                toCenter = toCenter / distToCenter;
            } else {
                toCenter = float2(0.0);
            }

            // Chromatic aberration - subtle for clarity
            float displacement = edgeFactor * uniforms.refractionStrength * edgeBand;
            float chromeStrength = edgeFactor * 3.0;  // Subtle RGB separation

            float2 redOffset = toCenter * (displacement + chromeStrength);
            float2 greenOffset = toCenter * displacement;
            float2 blueOffset = toCenter * (displacement - chromeStrength);

            float2 redUV = (uniforms.offset + magnifiedPixelPos + redOffset) / uniforms.backgroundSize;
            float2 greenUV = (uniforms.offset + magnifiedPixelPos + greenOffset) / uniforms.backgroundSize;
            float2 blueUV = (uniforms.offset + magnifiedPixelPos + blueOffset) / uniforms.backgroundSize;

            redUV = clamp(redUV, float2(0.001), float2(0.999));
            greenUV = clamp(greenUV, float2(0.001), float2(0.999));
            blueUV = clamp(blueUV, float2(0.001), float2(0.999));

            float r = backgroundTexture.sample(texSampler, redUV).r;
            float g = backgroundTexture.sample(texSampler, greenUV).g;
            float b = backgroundTexture.sample(texSampler, blueUV).b;
            float3 color = float3(r, g, b);

            float3 tint = uniforms.isDarkMode > 0.5
                ? float3(0.3, 0.3, 0.3)
                : float3(0.85, 0.85, 0.85);
            color = mix(color, tint, 0.02);

            float aa = 1.0 - smoothstep(-1.0, 0.5, sdf);
            return float4(color.rgb, aa);
        }
        """
}

private final class LensMetalView: UIView {
    private weak var sourceView: UIView?
    private var metalView: MTKView!
    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var pipelineState: MTLRenderPipelineState?
    private var displayLink: CADisplayLink?
    private var backgroundTexture: MTLTexture?
    private let ciContext: CIContext

    private var uniforms = LensUniforms()

    struct LensUniforms {
        var size: SIMD2<Float> = .zero
        var offset: SIMD2<Float> = .zero
        var backgroundSize: SIMD2<Float> = .zero
        var cornerRadius: Float = 0
        var refractionStrength: Float = 0.6  // Balanced for readability
        var magnificationStrength: Float = 0.12  // Subtle magnification - maintains clarity
        var liftAmount: Float = 0
        var isDarkMode: Float = 0
    }

    var cornerRadius: CGFloat = 32.0
    var liftAmount: CGFloat = 0.0
    var isDarkMode: Bool = false

    private var lastCaptureTime: CFTimeInterval = 0
    private let minCaptureInterval: CFTimeInterval = 1.0 / 20.0

    init(sourceView: UIView) {
        self.sourceView = sourceView

        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal not available")
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.ciContext = CIContext(mtlDevice: device, options: [.workingColorSpace: NSNull()])

        super.init(frame: .zero)

        setupMetal()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    deinit {
        displayLink?.invalidate()
    }

    private func setupMetal() {
        let mtkView = MTKView(frame: bounds, device: device)
        mtkView.backgroundColor = .clear
        mtkView.isOpaque = false
        mtkView.framebufferOnly = false
        mtkView.isPaused = false
        mtkView.preferredFramesPerSecond = 60
        mtkView.delegate = self
        self.metalView = mtkView
        addSubview(mtkView)

        guard let library = try? device.makeLibrary(source: LensShaderSource.code, options: nil),
              let vertexFunc = library.makeFunction(name: "lensVertex"),
              let fragmentFunc = library.makeFunction(name: "lensFragment") else {
            return
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunc
        descriptor.fragmentFunction = fragmentFunc
        descriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat
        descriptor.colorAttachments[0].isBlendingEnabled = true
        descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        descriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        pipelineState = try? device.makeRenderPipelineState(descriptor: descriptor)
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        if window != nil {
            startDisplayLink()
        } else {
            stopDisplayLink()
        }
    }

    private func startDisplayLink() {
        guard displayLink == nil else { return }
        let link = CADisplayLink(target: self, selector: #selector(updateFrame))
        link.preferredFramesPerSecond = 20
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    private func stopDisplayLink() {
        displayLink?.invalidate()
        displayLink = nil
    }

    @objc private func updateFrame() {
        let currentTime = CACurrentMediaTime()
        guard currentTime - lastCaptureTime >= minCaptureInterval else { return }
        lastCaptureTime = currentTime
        captureAndRender()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        metalView?.frame = bounds
    }

    private func captureAndRender() {
        guard let sourceView = sourceView else { return }
        guard bounds.width > 0, bounds.height > 0 else { return }

        let frameInSource = convert(bounds, to: sourceView)
        let padding: CGFloat = 10.0
        let captureRect = frameInSource.insetBy(dx: -padding, dy: -padding).intersection(sourceView.bounds)

        guard captureRect.width > 0, captureRect.height > 0 else { return }

        let scale: CGFloat = 1.0
        UIGraphicsBeginImageContextWithOptions(captureRect.size, true, scale)
        defer { UIGraphicsEndImageContext() }

        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        ctx.translateBy(x: -captureRect.origin.x, y: -captureRect.origin.y)

        let savedAlpha = alpha
        alpha = 0
        sourceView.layer.render(in: ctx)
        alpha = savedAlpha

        guard let snapshot = UIGraphicsGetImageFromCurrentImageContext(),
              let cgImage = snapshot.cgImage else { return }

        // NO BLUR - keep background sharp for clear refraction effect
        let textureLoader = MTKTextureLoader(device: device)
        backgroundTexture = try? textureLoader.newTexture(cgImage: cgImage, options: [.SRGB: false])

        let screenScale = UIScreen.main.scale
        let offset = CGPoint(x: frameInSource.origin.x - captureRect.origin.x, y: frameInSource.origin.y - captureRect.origin.y)

        uniforms.size = SIMD2<Float>(Float(bounds.width * screenScale), Float(bounds.height * screenScale))
        uniforms.offset = SIMD2<Float>(Float(offset.x * screenScale), Float(offset.y * screenScale))
        uniforms.backgroundSize = SIMD2<Float>(Float(captureRect.width * scale * screenScale), Float(captureRect.height * scale * screenScale))
        uniforms.cornerRadius = Float(cornerRadius * screenScale)
        uniforms.liftAmount = Float(liftAmount)
        uniforms.isDarkMode = isDarkMode ? 1.0 : 0.0

        metalView.setNeedsDisplay()
    }

    func setSourceView(_ view: UIView) {
        self.sourceView = view
    }
}

extension LensMetalView: MTKViewDelegate {
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
        encoder.setFragmentBytes(&uniformsCopy, length: MemoryLayout<LensUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

private final class RestingBackgroundView: UIVisualEffectView {
    var isDark: Bool?

    static func colorMatrix(isDark: Bool) -> [Float32] {
        if isDark {
            return [1.082, -0.113, -0.011, 0.0, 0.135, -0.034, 1.003, -0.011, 0.0, 0.135, -0.034, -0.113, 1.105, 0.0, 0.135, 0.0, 0.0, 0.0, 1.0, 0.0]
        } else {
            return [1.185, -0.05, -0.005, 0.0, -0.2, -0.015, 1.15, -0.005, 0.0, -0.2, -0.015, -0.05, 1.195, 0.0, -0.2, 0.0, 0.0, 0.0, 1.0, 0.0]
        }
    }

    init() {
        let effect = UIBlurEffect(style: .light)
        super.init(effect: effect)
        
        for subview in self.subviews {
            if subview.description.contains("VisualEffectSubview") {
                subview.isHidden = true
            }
        }
        
        self.clipsToBounds = true
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func update(isDark: Bool) {
        if self.isDark == isDark {
            return
        }
        self.isDark = isDark
        
        if let sublayer = self.layer.sublayers?[0], let _ = sublayer.filters {
            sublayer.backgroundColor = nil
            sublayer.isOpaque = false
            
            if let classValue = NSClassFromString("CAFilter") as AnyObject as? NSObjectProtocol {
                let makeSelector = NSSelectorFromString("filterWithName:")
                let filter = classValue.perform(makeSelector, with: "colorMatrix").takeUnretainedValue() as? NSObject
                
                if let filter {
                    var matrix: [Float32] = RestingBackgroundView.colorMatrix(isDark: isDark)
                    filter.setValue(NSValue(bytes: &matrix, objCType: "{CAColorMatrix=ffffffffffffffffffff}"), forKey: "inputColorMatrix")
                    sublayer.filters = [filter]
                    sublayer.setValue(1.0, forKey: "scale")
                }
            }
        }
    }
}

public final class LiquidLensView: UIView {
    private struct Params: Equatable {
        var size: CGSize
        var selectionX: CGFloat
        var selectionWidth: CGFloat
        var isDark: Bool
        var isLifted: Bool

        init(size: CGSize, selectionX: CGFloat, selectionWidth: CGFloat, isDark: Bool, isLifted: Bool) {
            self.size = size
            self.selectionX = selectionX
            self.selectionWidth = selectionWidth
            self.isLifted = isLifted
            self.isDark = isDark
        }
    }

    private struct LensParams: Equatable {
        var baseFrame: CGRect
        var isLifted: Bool

        init(baseFrame: CGRect, isLifted: Bool) {
            self.baseFrame = baseFrame
            self.isLifted = isLifted
        }
    }

    private let containerView: UIView
    private let backgroundContainerContainer: UIView
    private let backgroundContainer: GlassBackgroundContainerView
    private let backgroundView: GlassBackgroundView
    private var lensView: UIView?
    private var lensMetalView: LensMetalView?
    private let liftedContainerView: UIView
    public let contentView: UIView
    private let restingBackgroundView: RestingBackgroundView

    private var legacySelectionView: GlassBackgroundView.ContentImageView?
    private var legacyContentMaskView: UIView?
    private var legacyContentMaskBlobView: UIImageView?
    private var legacyLiftedContentBlobMaskView: UIImageView?

    public var selectedContentView: UIView {
        return self.liftedContainerView
    }

    private var params: Params?
    private var appliedLensParams: LensParams?
    private var isApplyingLensParams: Bool = false
    private var pendingLensParams: LensParams?

    private var liftedDisplayLink: SharedDisplayLinkDriver.Link?

    public var selectionX: CGFloat? {
        return self.params?.selectionX
    }

    public var selectionWidth: CGFloat? {
        return self.params?.selectionWidth
    }

    override public init(frame: CGRect) {
        self.containerView = UIView()
        
        self.backgroundContainerContainer = UIView()
        self.backgroundContainer = GlassBackgroundContainerView()
        
        self.backgroundView = GlassBackgroundView()
        
        self.contentView = UIView()
        self.liftedContainerView = UIView()

        self.restingBackgroundView = RestingBackgroundView()

        super.init(frame: frame)
        
        self.backgroundContainerContainer.addSubview(self.backgroundContainer)
        self.addSubview(self.backgroundContainerContainer)
        
        self.backgroundContainer.contentView.addSubview(self.backgroundView)
        self.backgroundView.contentView.addSubview(self.containerView)
        self.containerView.isUserInteractionEnabled = false
        
        if #available(iOS 26.0, *) {
            if let viewClass = NSClassFromString("_UILiquidLensView") as AnyObject as? NSObjectProtocol {
                let allocSelector = NSSelectorFromString("alloc")
                let initSelector = NSSelectorFromString("initWithRestingBackground:")
                let objcAlloc = viewClass.perform(allocSelector).takeUnretainedValue()
                let instance = objcAlloc.perform(initSelector, with: UIView()).takeUnretainedValue()
                self.lensView = instance as? UIView
            }
        }
        
        if let lensView = self.lensView {
            self.backgroundContainer.layer.zPosition = 1
            lensView.layer.zPosition = 10.0
            
            self.liftedContainerView.addSubview(self.restingBackgroundView)
            
            self.containerView.addSubview(self.liftedContainerView)
            self.containerView.addSubview(lensView)
            self.containerView.addSubview(self.contentView)
            
            lensView.perform(NSSelectorFromString("setLiftedContainerView:"), with: self.backgroundContainer.contentView)
            lensView.perform(NSSelectorFromString("setLiftedContentView:"), with: self.liftedContainerView)
            lensView.perform(NSSelectorFromString("setOverridePunchoutView:"), with: self.contentView)
            
            do {
                let selector = NSSelectorFromString("setLiftedContentMode:")
                if let method = lensView.method(for: selector) {
                    typealias ObjCMethod = @convention(c) (AnyObject, Selector, Int32) -> Void
                    let function = unsafeBitCast(method, to: ObjCMethod.self)
                    function(lensView, selector, 1)
                }
            }
            
            do {
                let selector = NSSelectorFromString("setStyle:")
                if let method = lensView.method(for: selector) {
                    typealias ObjCMethod = @convention(c) (AnyObject, Selector, Int32) -> Void
                    let function = unsafeBitCast(method, to: ObjCMethod.self)
                    function(lensView, selector, 1)
                }
            }
            
            do {
                let selector = NSSelectorFromString("setWarpsContentBelow:")
                if let method = lensView.method(for: selector) {
                    typealias ObjCMethod = @convention(c) (AnyObject, Selector, Bool) -> Void
                    let function = unsafeBitCast(method, to: ObjCMethod.self)
                    function(lensView, selector, true)
                }
            }
            
            lensView.setValue(UIColor(white: 0.0, alpha: 0.1), forKey: "restingBackgroundColor")
        } else {
            // Custom Metal lens for iOS 13-25
            var sourceView: UIView = self
            if let window = self.window {
                sourceView = window
            } else if let superview = self.superview {
                sourceView = superview.window ?? superview
            }

            let metalLens = LensMetalView(sourceView: sourceView)
            self.lensMetalView = metalLens
            metalLens.layer.zPosition = 5.0
            metalLens.alpha = 0.0  // Hidden initially, shown when lifted
            self.containerView.addSubview(metalLens)

            // Keep legacy selection view for non-lifted state
            let legacySelectionView = GlassBackgroundView.ContentImageView()
            self.legacySelectionView = legacySelectionView
            self.backgroundView.contentView.insertSubview(legacySelectionView, at: 0)

            let legacyContentMaskView = UIView()
            legacyContentMaskView.backgroundColor = .white
            self.legacyContentMaskView = legacyContentMaskView
            self.contentView.mask = legacyContentMaskView

            if let filter = CALayer.luminanceToAlpha() {
                legacyContentMaskView.layer.filters = [filter]
            }

            let legacyContentMaskBlobView = UIImageView()
            self.legacyContentMaskBlobView = legacyContentMaskBlobView
            legacyContentMaskView.addSubview(legacyContentMaskBlobView)

            self.containerView.addSubview(self.contentView)

            let legacyLiftedContentBlobMaskView = UIImageView()
            self.legacyLiftedContentBlobMaskView = legacyLiftedContentBlobMaskView
            self.liftedContainerView.mask = legacyLiftedContentBlobMaskView

            self.containerView.addSubview(self.liftedContainerView)
        }
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    public func update(size: CGSize, selectionX: CGFloat, selectionWidth: CGFloat, isDark: Bool, isLifted: Bool, transition: ComponentTransition) {
        let params = Params(size: size, selectionX: selectionX, selectionWidth: selectionWidth, isDark: isDark, isLifted: isLifted)
        if self.params == params {
            return
        }
        self.update(params: params, transition: transition)
    }

    private func update(transition: ComponentTransition) {
        guard let params = self.params else {
            return
        }
        self.update(params: params, transition: transition)
    }

    private func updateLens(params: LensParams, animated: Bool) {
        guard let lensView = self.lensView else {
            return
        }

        if self.isApplyingLensParams {
            self.pendingLensParams = params
            return
        }
        self.isApplyingLensParams = true
        let previousParams = self.appliedLensParams

        let transition: ComponentTransition = animated ? .easeInOut(duration: 0.3) : .immediate

        if previousParams?.isLifted != params.isLifted {
            let selector = NSSelectorFromString("setLifted:animated:alongsideAnimations:completion:")
            var shouldScheduleUpdate = false
            var didProcessUpdate = false
            self.pendingLensParams = params
            if let lensView = self.lensView, let method = lensView.method(for: selector) {
                typealias ObjCMethod = @convention(c) (AnyObject, Selector, Bool, Bool, @escaping () -> Void, AnyObject?) -> Void
                let function = unsafeBitCast(method, to: ObjCMethod.self)
                function(lensView, selector, params.isLifted, !transition.animation.isImmediate, { [weak self] in
                    guard let self else {
                        return
                    }
                    let liftedInset: CGFloat = params.isLifted ? 4.0 : -4.0
                    lensView.bounds = CGRect(origin: CGPoint(), size: CGSize(width: params.baseFrame.width + liftedInset * 2.0, height: params.baseFrame.height + liftedInset * 2.0))
                    didProcessUpdate = true
                    if shouldScheduleUpdate {
                        DispatchQueue.main.async { [weak self] in
                            guard let self, let pendingLensParams = self.pendingLensParams else {
                                return
                            }
                            self.isApplyingLensParams = false
                            self.pendingLensParams = nil
                            self.updateLens(params: pendingLensParams, animated: !transition.animation.isImmediate)
                        }
                    }
                }, nil)
            }
            if didProcessUpdate {
                transition.animateView {
                    lensView.center = CGPoint(x: params.baseFrame.midX, y: params.baseFrame.midY)
                }
                self.pendingLensParams = nil
                self.isApplyingLensParams = false
            } else {
                shouldScheduleUpdate = true
            }
        } else {
            transition.animateView {
                let liftedInset: CGFloat = params.isLifted ? 4.0 : -4.0
                lensView.bounds = CGRect(origin: CGPoint(), size: CGSize(width: params.baseFrame.width + liftedInset * 2.0, height: params.baseFrame.height + liftedInset * 2.0))
                lensView.center = CGPoint(x: params.baseFrame.midX, y: params.baseFrame.midY)
            }
            self.isApplyingLensParams = false
        }
    }

    private func updateLiftedLensPosition() {
        // Without this, the lens won't update its bouncing animations unless it's being moved
        if self.isApplyingLensParams {
            return
        }
        guard let lensView = self.lensView else {
            return
        }
        guard let params = self.appliedLensParams else {
            return
        }
        lensView.center = CGPoint(x: params.baseFrame.midX, y: params.baseFrame.midY)
    }

    private func update(params: Params, transition: ComponentTransition) {
        let isFirstTime = self.params == nil
        let transition: ComponentTransition = isFirstTime ? .immediate : transition

        self.params = params

        transition.setFrame(view: self.containerView, frame: CGRect(origin: CGPoint(), size: params.size))
        transition.setFrame(view: self.backgroundContainerContainer, frame: CGRect(origin: CGPoint(), size: params.size))

        transition.setFrame(view: self.backgroundContainer, frame: CGRect(origin: CGPoint(), size: params.size))
        self.backgroundContainer.update(size: params.size, isDark: params.isDark, transition: transition)
        
        transition.setFrame(view: self.backgroundView, frame: CGRect(origin: CGPoint(), size: params.size))
        self.backgroundView.update(size: params.size, cornerRadius: params.size.height * 0.5, isDark: params.isDark, tintColor: GlassBackgroundView.TintColor.init(kind: .panel, color: UIColor(white: params.isDark ? 0.0 : 1.0, alpha: 0.6)), isInteractive: true, transition: transition)
        
        transition.setFrame(view: self.contentView, frame: CGRect(origin: CGPoint(), size: params.size))
        transition.setFrame(view: self.liftedContainerView, frame: CGRect(origin: CGPoint(), size: params.size))

        let baseLensFrame = CGRect(origin: CGPoint(x: max(0.0, min(params.selectionX, params.size.width - params.selectionWidth)), y: 0.0), size: CGSize(width: params.selectionWidth, height: params.size.height))
        self.updateLens(params: LensParams(baseFrame: baseLensFrame, isLifted: params.isLifted), animated: !transition.animation.isImmediate)
        
        if let legacyContentMaskView = self.legacyContentMaskView {
            transition.setFrame(view: legacyContentMaskView, frame: CGRect(origin: CGPoint(), size: params.size))
        }
        if let metalLens = self.lensMetalView {
            let lensFrame = baseLensFrame.insetBy(dx: 4.0, dy: 4.0)
            let liftedInset: CGFloat = params.isLifted ? 4.0 : 0.0
            let effectiveLensFrame = lensFrame.insetBy(dx: -liftedInset, dy: -liftedInset)

            transition.setFrame(view: metalLens, frame: effectiveLensFrame)
            metalLens.cornerRadius = effectiveLensFrame.height * 0.5
            metalLens.isDarkMode = params.isDark
            metalLens.liftAmount = params.isLifted ? 1.0 : 0.0

            // Only show when lifted
            transition.setAlpha(view: metalLens, alpha: params.isLifted ? 1.0 : 0.0)

            // Update source view if window is now available
            if let window = self.window {
                metalLens.setSourceView(window)
            }
        }

        if let legacyContentMaskBlobView = self.legacyContentMaskBlobView, let legacyLiftedContentBlobMaskView = self.legacyLiftedContentBlobMaskView, let legacySelectionView = self.legacySelectionView {
            let lensFrame = baseLensFrame.insetBy(dx: 4.0, dy: 4.0)
            let effectiveLensFrame = lensFrame.insetBy(dx: params.isLifted ? -2.0 : 0.0, dy: params.isLifted ? -2.0 : 0.0)

            if legacyContentMaskBlobView.image?.size.height != lensFrame.height {
                legacyContentMaskBlobView.image = generateStretchableFilledCircleImage(diameter: lensFrame.height, color: .black)
                legacyLiftedContentBlobMaskView.image = legacyContentMaskBlobView.image
                legacySelectionView.image = generateStretchableFilledCircleImage(diameter: lensFrame.height, color: .white)?.withRenderingMode(.alwaysTemplate)
            }
            transition.setFrame(view: legacyContentMaskBlobView, frame: effectiveLensFrame)
            transition.setFrame(view: legacyLiftedContentBlobMaskView, frame: effectiveLensFrame)

            legacySelectionView.tintColor = UIColor(white: params.isDark ? 1.0 : 0.0, alpha: params.isDark ? 0.1 : 0.075)
            transition.setFrame(view: legacySelectionView, frame: effectiveLensFrame)
        }

        transition.setFrame(view: self.restingBackgroundView, frame: CGRect(origin: CGPoint(), size: params.size))
        self.restingBackgroundView.update(isDark: params.isDark)
        transition.setAlpha(view: self.restingBackgroundView, alpha: params.isLifted ? 0.0 : 1.0)

        if params.isLifted {
            if self.liftedDisplayLink == nil {
                self.liftedDisplayLink = SharedDisplayLinkDriver.shared.add(framesPerSecond: .max, { [weak self] _ in
                    guard let self else {
                        return
                    }
                    self.updateLiftedLensPosition()
                })
            }
        } else if let liftedDisplayLink = self.liftedDisplayLink {
            self.liftedDisplayLink = nil
            liftedDisplayLink.invalidate()
        }
    }
}
