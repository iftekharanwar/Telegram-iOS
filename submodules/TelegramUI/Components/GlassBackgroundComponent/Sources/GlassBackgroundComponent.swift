import Foundation
import UIKit
import Display
import ComponentFlow
import ComponentDisplayAdapters
import UIKitRuntimeUtils
import CoreImage
import AppBundle
import MetalKit

// MARK: - Advanced Material Rendering System

private struct RefractionShaderDescriptor {
    static let source = """
        #include <metal_stdlib>
        using namespace metal;

        struct RasterizationData {
            float4 clipSpacePosition [[position]];
            float2 textureCoordinate;
        };

        struct MaterialParams {
            float2 dimensions;
            float2 contentOffset;
            float2 sourceSize;
            float radiusValue;
            float aberrationIntensity;
            float appearanceMode;
        };

        vertex RasterizationData vertexTransform(uint vertexID [[vertex_id]]) {
            constexpr float2 positions[] = {
                float2(-1.0, -1.0), float2(1.0, -1.0),
                float2(-1.0, 1.0), float2(1.0, 1.0)
            };
            constexpr float2 coords[] = {
                float2(0.0, 1.0), float2(1.0, 1.0),
                float2(0.0, 0.0), float2(1.0, 0.0)
            };
            return RasterizationData{float4(positions[vertexID], 0.0, 1.0), coords[vertexID]};
        }

        float signedDistanceRoundRect(float2 samplePoint, float2 extent, float radius) {
            float2 delta = abs(samplePoint) - extent + radius;
            return min(max(delta.x, delta.y), 0.0) + length(max(delta, 0.0)) - radius;
        }

        fragment float4 applyMaterialEffect(
            RasterizationData in [[stage_in]],
            texture2d<float, access::sample> sourceTexture [[texture(0)]],
            constant MaterialParams &params [[buffer(0)]]
        ) {
            constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);

            float2 normalizedCoord = in.textureCoordinate;
            float2 pixelCoord = normalizedCoord * params.dimensions;
            float2 centerPoint = params.dimensions * 0.5;
            float2 centerOffset = pixelCoord - centerPoint;

            float distanceField = signedDistanceRoundRect(
                centerOffset,
                float2(params.dimensions.x * 0.5, params.dimensions.y * 0.5),
                params.radiusValue
            );

            if (distanceField > 0.0) discard_fragment();

            float edgeProximity = -distanceField;
            float influenceRadius = min(params.dimensions.x, params.dimensions.y) * 0.3;
            float edgeInfluence = 1.0 - smoothstep(0.0, influenceRadius, edgeProximity);
            edgeInfluence = pow(edgeInfluence, 3.0) * 2.0;

            float2 centerReference;
            float insetH = min(params.radiusValue, params.dimensions.x * 0.5);
            float insetV = min(params.radiusValue, params.dimensions.y * 0.5);

            if (params.dimensions.x >= params.dimensions.y) {
                centerReference = float2(clamp(pixelCoord.x, insetH, params.dimensions.x - insetH), centerPoint.y);
            } else {
                centerReference = float2(centerPoint.x, clamp(pixelCoord.y, insetV, params.dimensions.y - insetV));
            }

            float2 directionToCenter = centerReference - pixelCoord;
            float distanceToCenter = length(directionToCenter);
            directionToCenter = distanceToCenter > 0.001 ? directionToCenter / distanceToCenter : float2(0.0);

            float displacementMagnitude = edgeInfluence * params.aberrationIntensity * influenceRadius;
            float chromaticShift = edgeInfluence * 3.0;  // Subtle RGB separation for clarity

            float2 redShift = directionToCenter * (displacementMagnitude + chromaticShift);
            float2 greenShift = directionToCenter * displacementMagnitude;
            float2 blueShift = directionToCenter * (displacementMagnitude - chromaticShift);

            float2 redCoord = (params.contentOffset + pixelCoord + redShift) / params.sourceSize;
            float2 greenCoord = (params.contentOffset + pixelCoord + greenShift) / params.sourceSize;
            float2 blueCoord = (params.contentOffset + pixelCoord + blueShift) / params.sourceSize;

            redCoord = clamp(redCoord, float2(0.001), float2(0.999));
            greenCoord = clamp(greenCoord, float2(0.001), float2(0.999));
            blueCoord = clamp(blueCoord, float2(0.001), float2(0.999));

            float3 sampledColor = float3(
                sourceTexture.sample(linearSampler, redCoord).r,
                sourceTexture.sample(linearSampler, greenCoord).g,
                sourceTexture.sample(linearSampler, blueCoord).b
            );

            float3 tintValue = params.appearanceMode > 0.5 ? float3(0.3) : float3(0.85);
            sampledColor = mix(sampledColor, tintValue, 0.02);

            float edgeAntialias = 1.0 - smoothstep(-1.0, 0.5, distanceField);

            return float4(sampledColor, edgeAntialias);
        }
        """
}

private final class AdvancedMaterialView: UIView {
    private final class RenderingContext {
        let device: MTLDevice
        let commandQueue: MTLCommandQueue
        let pipelineState: MTLRenderPipelineState
        var activeTexture: MTLTexture?

        struct Parameters {
            var dimensions: SIMD2<Float> = .zero
            var contentOffset: SIMD2<Float> = .zero
            var sourceSize: SIMD2<Float> = .zero
            var radiusValue: Float = 0
            var aberrationIntensity: Float = 0
            var appearanceMode: Float = 0
        }

        var parameters = Parameters()

        init?(device: MTLDevice, pixelFormat: MTLPixelFormat) {
            self.device = device
            guard let queue = device.makeCommandQueue() else {
                print("⚠️ RenderingContext: Failed to create command queue")
                return nil
            }
            self.commandQueue = queue

            do {
                let library = try device.makeLibrary(source: RefractionShaderDescriptor.source, options: nil)
                guard let vertexFunc = library.makeFunction(name: "vertexTransform"),
                      let fragmentFunc = library.makeFunction(name: "applyMaterialEffect") else {
                    print("⚠️ RenderingContext: Failed to find shader functions")
                    return nil
                }

                let pipelineDescriptor = MTLRenderPipelineDescriptor()
                pipelineDescriptor.vertexFunction = vertexFunc
                pipelineDescriptor.fragmentFunction = fragmentFunc
                pipelineDescriptor.colorAttachments[0].pixelFormat = pixelFormat
                pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
                pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
                pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
                pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
                pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

                let pipeline = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
                self.pipelineState = pipeline
                print("✅ RenderingContext: Pipeline created successfully")
            } catch {
                print("⚠️ RenderingContext: Shader compilation failed: \(error)")
                return nil
            }
        }

        func encode(into encoder: MTLRenderCommandEncoder) {
            guard let texture = activeTexture else { return }
            encoder.setRenderPipelineState(pipelineState)
            encoder.setFragmentTexture(texture, index: 0)
            var params = parameters
            encoder.setFragmentBytes(&params, length: MemoryLayout<Parameters>.stride, index: 0)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
    }

    private final class BackgroundCaptureEngine {
        private let imageContext: CIContext
        private weak var targetView: UIView?
        private var lastUpdateTime: CFTimeInterval = 0

        init(targetView: UIView) {
            self.targetView = targetView
            self.imageContext = CIContext(options: [.useSoftwareRenderer: false])
        }

        func captureBackground(for bounds: CGRect, scale: CGFloat, padding: CGFloat, hiddenView: UIView?) -> (image: UIImage, offset: CGPoint)? {
            guard let target = targetView else { return nil }
            guard bounds.width > 0, bounds.height > 0 else { return nil }

            let frameInTarget = hiddenView?.convert(bounds, to: target) ?? bounds
            let captureRegion = frameInTarget.insetBy(dx: -padding, dy: -padding).intersection(target.bounds)

            guard captureRegion.width > 0, captureRegion.height > 0 else { return nil }

            UIGraphicsBeginImageContextWithOptions(captureRegion.size, false, scale)
            defer { UIGraphicsEndImageContext() }

            guard let context = UIGraphicsGetCurrentContext() else { return nil }
            context.translateBy(x: -captureRegion.origin.x, y: -captureRegion.origin.y)

            var parentGlassView: UIView? = hiddenView?.superview
            while let parent = parentGlassView {
                if parent is GlassBackgroundView {
                    break
                }
                parentGlassView = parent.superview
            }

            let savedHiddenAlpha = hiddenView?.alpha ?? 1.0
            let savedParentAlpha = parentGlassView?.alpha ?? 1.0

            hiddenView?.alpha = 0
            parentGlassView?.alpha = 0

            target.layer.render(in: context)

            hiddenView?.alpha = savedHiddenAlpha
            parentGlassView?.alpha = savedParentAlpha

            guard let snapshot = UIGraphicsGetImageFromCurrentImageContext(),
                  let cgImage = snapshot.cgImage else { return nil }

            let ciImage = CIImage(cgImage: cgImage)
            let blurFilter = CIFilter(name: "CIGaussianBlur")
            blurFilter?.setValue(ciImage, forKey: kCIInputImageKey)
            blurFilter?.setValue(2.0, forKey: kCIInputRadiusKey)

            if let blurredCIImage = blurFilter?.outputImage {
                let ciContext = CIContext(options: [.useSoftwareRenderer: false])
                if let blurredCGImage = ciContext.createCGImage(blurredCIImage, from: blurredCIImage.extent) {
                    let finalImage = UIImage(cgImage: blurredCGImage, scale: scale, orientation: .up)
                    let offset = CGPoint(
                        x: frameInTarget.origin.x - captureRegion.origin.x,
                        y: frameInTarget.origin.y - captureRegion.origin.y
                    )
                    return (finalImage, offset)
                }
            }

            // Fallback to non-blurred if blur fails
            let finalImage = UIImage(cgImage: cgImage, scale: scale, orientation: .up)
            let offset = CGPoint(
                x: frameInTarget.origin.x - captureRegion.origin.x,
                y: frameInTarget.origin.y - captureRegion.origin.y
            )
            return (finalImage, offset)
        }
    }

    fileprivate weak var contentSourceView: UIView?
    private var metalDisplayView: MTKView!
    private var renderContext: RenderingContext?
    private var updateTimer: CADisplayLink?
    private let illuminationLayer: CAGradientLayer
    private var captureEngine: BackgroundCaptureEngine?

    private var lastRenderTimestamp: CFTimeInterval = 0
    private let minimumUpdateInterval: CFTimeInterval = 1.0 / 60.0

    var shapeRadius: CGFloat = 32 {
        didSet {
            guard shapeRadius != oldValue else { return }
            scheduleUpdate()
        }
    }

    var distortionStrength: Float = 0.6  // Balanced - visible effect but maintains readability
    var isDarkAppearance: Bool = false {
        didSet {
            guard isDarkAppearance != oldValue else { return }
            scheduleUpdate()
        }
    }

    init(contentSource: UIView) {
        self.contentSourceView = contentSource
        self.illuminationLayer = CAGradientLayer()
        super.init(frame: .zero)
        configureComponents()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    deinit {
        updateTimer?.invalidate()
    }

    func updateContentSource(_ view: UIView) {
        contentSourceView = view
        captureEngine = BackgroundCaptureEngine(targetView: view)
        scheduleUpdate()
    }

    private func configureComponents() {
        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            print("⚠️ AdvancedMaterialView: Metal device not available")
            return
        }

        illuminationLayer.colors = [
            UIColor(white: 1.0, alpha: 0.03).cgColor,
            UIColor(white: 1.0, alpha: 0.0).cgColor
        ]
        illuminationLayer.locations = [0.0, 1.0]
        layer.addSublayer(illuminationLayer)

        metalDisplayView = MTKView()
        metalDisplayView.device = metalDevice
        metalDisplayView.backgroundColor = .clear
        metalDisplayView.isOpaque = false
        metalDisplayView.framebufferOnly = false
        metalDisplayView.isPaused = false
        metalDisplayView.enableSetNeedsDisplay = false
        metalDisplayView.preferredFramesPerSecond = 60
        addSubview(metalDisplayView)

        guard let context = RenderingContext(device: metalDevice, pixelFormat: metalDisplayView.colorPixelFormat) else {
            print("⚠️ AdvancedMaterialView: Failed to create RenderingContext")
            return
        }
        renderContext = context
        metalDisplayView.delegate = self

        if let source = contentSourceView {
            captureEngine = BackgroundCaptureEngine(targetView: source)
        }

        print("✅ AdvancedMaterialView: Configured successfully")
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        if window != nil {
            activateUpdateTimer()
            performBackgroundCapture()
        } else {
            deactivateUpdateTimer()
        }
    }

    private func activateUpdateTimer() {
        guard updateTimer == nil else { return }
        let timer = CADisplayLink(target: self, selector: #selector(timerTriggered))
        timer.preferredFramesPerSecond = 20
        timer.add(to: .main, forMode: .common)
        updateTimer = timer
    }

    private func deactivateUpdateTimer() {
        updateTimer?.invalidate()
        updateTimer = nil
    }

    @objc private func timerTriggered() {
        let timestamp = CACurrentMediaTime()
        guard timestamp - lastRenderTimestamp >= minimumUpdateInterval else { return }
        lastRenderTimestamp = timestamp
        performBackgroundCapture()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        metalDisplayView?.frame = bounds
        illuminationLayer.frame = bounds
        illuminationLayer.cornerRadius = shapeRadius
        scheduleUpdate()
    }

    private func scheduleUpdate() {
        // Trigger capture on next cycle
    }

    private func performBackgroundCapture() {
        guard let engine = captureEngine, let context = renderContext else { return }
        guard bounds.width > 0, bounds.height > 0 else { return }

        guard let result = engine.captureBackground(
            for: bounds,
            scale: 1.0,
            padding: 10.0,
            hiddenView: self
        ) else { return }

        let screenScale = UIScreen.main.scale
        context.parameters.dimensions = SIMD2<Float>(
            Float(bounds.width * screenScale),
            Float(bounds.height * screenScale)
        )
        context.parameters.contentOffset = SIMD2<Float>(
            Float(result.offset.x * screenScale),
            Float(result.offset.y * screenScale)
        )
        context.parameters.sourceSize = SIMD2<Float>(
            Float(result.image.size.width * screenScale),
            Float(result.image.size.height * screenScale)
        )
        context.parameters.radiusValue = Float(shapeRadius * screenScale)
        context.parameters.aberrationIntensity = distortionStrength
        context.parameters.appearanceMode = isDarkAppearance ? 1.0 : 0.0

        if let cgImage = result.image.cgImage {
            let loader = MTKTextureLoader(device: context.device)
            context.activeTexture = try? loader.newTexture(cgImage: cgImage, options: [.SRGB: false])
        }

        metalDisplayView.setNeedsDisplay()
    }
}

extension AdvancedMaterialView: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Size changes handled in layoutSubviews
    }

    func draw(in view: MTKView) {
        guard let context = renderContext,
              let drawable = view.currentDrawable,
              let passDescriptor = view.currentRenderPassDescriptor else { return }

        passDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        passDescriptor.colorAttachments[0].loadAction = .clear

        guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) else { return }

        context.encode(into: renderEncoder)
        renderEncoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

// MARK: - Content Container

private final class ContentContainer: UIView {
    private let maskContentView: UIView

    init(maskContentView: UIView) {
        self.maskContentView = maskContentView

        super.init(frame: CGRect())
    }

    required public init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func hitTest(_ point: CGPoint, with event: UIEvent?) -> UIView? {
        guard let result = super.hitTest(point, with: event) else {
            return nil
        }
        if result === self {
            if let gestureRecognizers = self.gestureRecognizers, !gestureRecognizers.isEmpty {
                return result
            }
            return nil
        }
        return result
    }
    
    override func didAddSubview(_ subview: UIView) {
        super.didAddSubview(subview)
        
        if let subview = subview as? GlassBackgroundView.ContentView {
            self.maskContentView.addSubview(subview.tintMask)
        }
    }
    
    override func willRemoveSubview(_ subview: UIView) {
        super.willRemoveSubview(subview)
        
        if let subview = subview as? GlassBackgroundView.ContentView {
            subview.tintMask.removeFromSuperview()
        }
    }
}

public class GlassBackgroundView: UIView {
    public protocol ContentView: UIView {
        var tintMask: UIView { get }
    }
    
    open class ContentLayer: SimpleLayer {
        public var targetLayer: CALayer?
        
        override init() {
            super.init()
        }
        
        override init(layer: Any) {
            super.init(layer: layer)
        }
        
        required public init?(coder: NSCoder) {
            fatalError("init(coder:) has not been implemented")
        }
        
        override public var position: CGPoint {
            get {
                return super.position
            } set(value) {
                if let targetLayer = self.targetLayer {
                    targetLayer.position = value
                }
                super.position = value
            }
        }
        
        override public var bounds: CGRect {
            get {
                return super.bounds
            } set(value) {
                if let targetLayer = self.targetLayer {
                    targetLayer.bounds = value
                }
                super.bounds = value
            }
        }
        
        override public var anchorPoint: CGPoint {
            get {
                return super.anchorPoint
            } set(value) {
                if let targetLayer = self.targetLayer {
                    targetLayer.anchorPoint = value
                }
                super.anchorPoint = value
            }
        }
        
        override public var anchorPointZ: CGFloat {
            get {
                return super.anchorPointZ
            } set(value) {
                if let targetLayer = self.targetLayer {
                    targetLayer.anchorPointZ = value
                }
                super.anchorPointZ = value
            }
        }
        
        override public var opacity: Float {
            get {
                return super.opacity
            } set(value) {
                if let targetLayer = self.targetLayer {
                    targetLayer.opacity = value
                }
                super.opacity = value
            }
        }
        
        override public var sublayerTransform: CATransform3D {
            get {
                return super.sublayerTransform
            } set(value) {
                if let targetLayer = self.targetLayer {
                    targetLayer.sublayerTransform = value
                }
                super.sublayerTransform = value
            }
        }
        
        override public var transform: CATransform3D {
            get {
                return super.transform
            } set(value) {
                if let targetLayer = self.targetLayer {
                    targetLayer.transform = value
                }
                super.transform = value
            }
        }
        
        override public func add(_ animation: CAAnimation, forKey key: String?) {
            if let targetLayer = self.targetLayer {
                targetLayer.add(animation, forKey: key)
            }
            
            super.add(animation, forKey: key)
        }
        
        override public func removeAllAnimations() {
            if let targetLayer = self.targetLayer {
                targetLayer.removeAllAnimations()
            }
            
            super.removeAllAnimations()
        }
        
        override public func removeAnimation(forKey: String) {
            if let targetLayer = self.targetLayer {
                targetLayer.removeAnimation(forKey: forKey)
            }
            
            super.removeAnimation(forKey: forKey)
        }
    }
    
    public final class ContentColorView: UIView, ContentView {
        override public static var layerClass: AnyClass {
            return ContentLayer.self
        }
        
        public let tintMask: UIView
        
        override public init(frame: CGRect) {
            self.tintMask = UIView()
            
            super.init(frame: CGRect())
            
            self.tintMask.tintColor = .black
        }
        
        required public init?(coder: NSCoder) {
            fatalError("init(coder:) has not been implemented")
        }
    }
    
    public final class ContentImageView: UIImageView, ContentView {
        override public static var layerClass: AnyClass {
            return ContentLayer.self
        }
        
        private let tintImageView: UIImageView
        public var tintMask: UIView {
            return self.tintImageView
        }
        
        override public var image: UIImage? {
            didSet {
                self.tintImageView.image = self.image
            }
        }
        
        override public var tintColor: UIColor? {
            didSet {
                if self.tintColor != oldValue {
                    self.setMonochromaticEffect(tintColor: self.tintColor)
                }
            }
        }
        
        override public init(frame: CGRect) {
            self.tintImageView = UIImageView()
            
            super.init(frame: CGRect())
            
            self.tintImageView.tintColor = .black
        }
        
        override public init(image: UIImage?) {
            self.tintImageView = UIImageView()
            
            super.init(image: image)
            
            self.tintImageView.image = image
            self.tintImageView.tintColor = .black
        }
        
        override public init(image: UIImage?, highlightedImage: UIImage?) {
            self.tintImageView = UIImageView()
            
            super.init(image: image, highlightedImage: highlightedImage)
            
            self.tintImageView.image = image
            self.tintImageView.tintColor = .black
        }
        
        required public init?(coder: NSCoder) {
            fatalError("init(coder:) has not been implemented")
        }
    }
    
    public struct TintColor: Equatable {
        public enum Kind {
            case panel
            case custom
        }
        
        public let kind: Kind
        public let color: UIColor
        public let innerColor: UIColor?
        
        public init(kind: Kind, color: UIColor, innerColor: UIColor? = nil) {
            self.kind = kind
            self.color = color
            self.innerColor = innerColor
        }
    }
    
    public enum Shape: Equatable {
        case roundedRect(cornerRadius: CGFloat)
    }
    
    private final class ClippingShapeContext {
        let view: UIView
        
        private(set) var shape: Shape?
        
        init(view: UIView) {
            self.view = view
        }
        
        func update(shape: Shape, size: CGSize, transition: ComponentTransition) {
            self.shape = shape
            
            switch shape {
            case let .roundedRect(cornerRadius):
                transition.setCornerRadius(layer: self.view.layer, cornerRadius: cornerRadius)
            }
        }
    }
    
    public struct Params: Equatable {
        public let shape: Shape
        public let isDark: Bool
        public let tintColor: TintColor
        public let isInteractive: Bool
        
        init(shape: Shape, isDark: Bool, tintColor: TintColor, isInteractive: Bool) {
            self.shape = shape
            self.isDark = isDark
            self.tintColor = tintColor
            self.isInteractive = isInteractive
        }
    }
    
    private let backgroundNode: NavigationBackgroundNode?
    private var materialView: AdvancedMaterialView?

    private let nativeView: UIVisualEffectView?
    private let nativeViewClippingContext: ClippingShapeContext?
    private let nativeParamsView: EffectSettingsContainerView?

    private let foregroundView: UIImageView?
    private let shadowView: UIImageView?

    private let maskContainerView: UIView
    public let maskContentView: UIView
    private let contentContainer: ContentContainer

    private var whiteBacklightView: UIView?
    private var innerBackgroundView: UIView?
    
    public var contentView: UIView {
        if let nativeView = self.nativeView {
            return nativeView.contentView
        } else {
            return self.contentContainer
        }
    }
    
    public private(set) var params: Params?
        
    public static var useCustomGlassImpl: Bool = true

    public override init(frame: CGRect) {
        if #available(iOS 26.0, *), !GlassBackgroundView.useCustomGlassImpl {
            self.backgroundNode = nil
            self.materialView = nil

            let glassEffect = UIGlassEffect(style: .regular)
            glassEffect.isInteractive = false
            let nativeView = UIVisualEffectView(effect: glassEffect)
            self.nativeViewClippingContext = ClippingShapeContext(view: nativeView)
            self.nativeView = nativeView

            let nativeParamsView = EffectSettingsContainerView(frame: CGRect())
            self.nativeParamsView = nativeParamsView

            nativeParamsView.addSubview(nativeView)

            self.foregroundView = nil
            self.shadowView = nil
        } else if GlassBackgroundView.useCustomGlassImpl {
            self.backgroundNode = nil
            self.nativeView = nil
            self.nativeViewClippingContext = nil
            self.nativeParamsView = nil
            self.foregroundView = nil
            self.shadowView = nil

            self.materialView = nil
        } else {
            let backgroundNode = NavigationBackgroundNode(color: .black, enableBlur: true, customBlurRadius: 8.0)
            self.backgroundNode = backgroundNode
            self.materialView = nil
            self.nativeView = nil
            self.nativeViewClippingContext = nil
            self.nativeParamsView = nil
            self.foregroundView = UIImageView()

            self.shadowView = UIImageView()
        }
        
        self.maskContainerView = UIView()
        self.maskContainerView.backgroundColor = .white
        if let filter = CALayer.luminanceToAlpha() {
            self.maskContainerView.layer.filters = [filter]
        }
        
        self.maskContentView = UIView()
        self.maskContainerView.addSubview(self.maskContentView)
        
        self.contentContainer = ContentContainer(maskContentView: self.maskContentView)
        
        super.init(frame: frame)
        
        if let shadowView = self.shadowView {
            self.addSubview(shadowView)
        }
        if let nativeParamsView = self.nativeParamsView {
            self.addSubview(nativeParamsView)
        }
        if let backgroundNode = self.backgroundNode {
            self.addSubview(backgroundNode.view)
        }
        if let foregroundView = self.foregroundView {
            self.addSubview(foregroundView)
            foregroundView.mask = self.maskContainerView
        }
        self.addSubview(self.contentContainer)
    }
    
    required public init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override public func didMoveToWindow() {
        super.didMoveToWindow()

        // Update content source when window becomes available
        if let materialView = self.materialView, let window = self.window {
            if materialView.contentSourceView !== window {
                materialView.updateContentSource(window)
            }
        }
    }

    override public func hitTest(_ point: CGPoint, with event: UIEvent?) -> UIView? {
        if let nativeView = self.nativeView {
            if let result = nativeView.hitTest(self.convert(point, to: nativeView), with: event) {
                return result
            }
        } else {
            if let result = self.contentContainer.hitTest(self.convert(point, to: self.contentContainer), with: event) {
                return result
            }
        }
        return nil
    }
        
    public func update(size: CGSize, cornerRadius: CGFloat, isDark: Bool, tintColor: TintColor, isInteractive: Bool = false, transition: ComponentTransition) {
        self.update(size: size, shape: .roundedRect(cornerRadius: cornerRadius), isDark: isDark, tintColor: tintColor, isInteractive: isInteractive, transition: transition)
    }
    
    public func update(size: CGSize, shape: Shape, isDark: Bool, tintColor: TintColor, isInteractive: Bool = false, transition: ComponentTransition) {
        if GlassBackgroundView.useCustomGlassImpl && self.backgroundNode == nil && self.nativeView == nil {
            if self.whiteBacklightView == nil {
                let backlightView = UIView()
                backlightView.backgroundColor = UIColor.white
                backlightView.alpha = 0.7
                self.whiteBacklightView = backlightView
                self.insertSubview(backlightView, at: 0)
            }

            if self.materialView == nil {
                var contentSource: UIView = self
                if let window = self.window {
                    contentSource = window
                } else if let superview = self.superview {
                    contentSource = superview.window ?? superview
                }

                let advancedView = AdvancedMaterialView(contentSource: contentSource)
                advancedView.alpha = 0.2
                self.materialView = advancedView
                if let backlightView = self.whiteBacklightView {
                    self.insertSubview(advancedView, aboveSubview: backlightView)
                } else {
                    self.insertSubview(advancedView, at: 0)
                }
            }

            if let backlightView = self.whiteBacklightView {
                switch shape {
                case let .roundedRect(cornerRadius):
                    backlightView.layer.cornerRadius = cornerRadius
                    backlightView.layer.masksToBounds = true
                }

                if transition.animation.isImmediate {
                    backlightView.frame = CGRect(origin: CGPoint(), size: size)
                } else {
                    transition.setFrame(view: backlightView, frame: CGRect(origin: CGPoint(), size: size))
                }
            }

            if let materialView = self.materialView {
                // Always try to update to window if available
                if let window = self.window, materialView.contentSourceView !== window {
                    materialView.updateContentSource(window)
                } else if materialView.contentSourceView == nil, let superview = self.superview?.window {
                    materialView.updateContentSource(superview)
                }

                // Update properties
                switch shape {
                case let .roundedRect(cornerRadius):
                    materialView.shapeRadius = cornerRadius
                }

                materialView.isDarkAppearance = isDark

                if transition.animation.isImmediate {
                    materialView.frame = CGRect(origin: CGPoint(), size: size)
                } else {
                    transition.setFrame(view: materialView, frame: CGRect(origin: CGPoint(), size: size))
                }
            }
        }

        if let nativeView = self.nativeView, let nativeViewClippingContext = self.nativeViewClippingContext, (nativeView.bounds.size != size || nativeViewClippingContext.shape != shape) {

            nativeViewClippingContext.update(shape: shape, size: size, transition: transition)
            if transition.animation.isImmediate {
                nativeView.frame = CGRect(origin: CGPoint(), size: size)
            } else {
                let nativeFrame = CGRect(origin: CGPoint(), size: size)
                transition.setFrame(view: nativeView, frame: nativeFrame)
            }
        }
        if let backgroundNode = self.backgroundNode {
            backgroundNode.updateColor(color: .clear, forceKeepBlur: tintColor.color.alpha != 1.0, transition: transition.containedViewLayoutTransition)

            switch shape {
            case let .roundedRect(cornerRadius):
                backgroundNode.update(size: size, cornerRadius: cornerRadius, transition: transition.containedViewLayoutTransition)
            }
            transition.setFrame(view: backgroundNode.view, frame: CGRect(origin: CGPoint(), size: size))
        }
        
        let shadowInset: CGFloat = 32.0
        
        if let innerColor = tintColor.innerColor {
            let innerBackgroundFrame = CGRect(origin: CGPoint(), size: size).insetBy(dx: 3.0, dy: 3.0)
            let innerBackgroundRadius = min(innerBackgroundFrame.width, innerBackgroundFrame.height) * 0.5
            
            let innerBackgroundView: UIView
            var innerBackgroundTransition = transition
            var animateIn = false
            if let current = self.innerBackgroundView {
                innerBackgroundView = current
            } else {
                innerBackgroundView = UIView()
                innerBackgroundTransition = innerBackgroundTransition.withAnimation(.none)
                self.innerBackgroundView = innerBackgroundView
                self.contentView.insertSubview(innerBackgroundView, at: 0)
                
                innerBackgroundView.frame = innerBackgroundFrame
                innerBackgroundView.layer.cornerRadius = innerBackgroundRadius
                animateIn = true
            }
            
            innerBackgroundView.backgroundColor = innerColor
            innerBackgroundTransition.setFrame(view: innerBackgroundView, frame: innerBackgroundFrame)
            innerBackgroundTransition.setCornerRadius(layer: innerBackgroundView.layer, cornerRadius: innerBackgroundRadius)
            
            if animateIn {
                transition.animateAlpha(view: innerBackgroundView, from: 0.0, to: 1.0)
                transition.animateScale(view: innerBackgroundView, from: 0.001, to: 1.0)
            }
        } else if let innerBackgroundView = self.innerBackgroundView {
            self.innerBackgroundView = nil
            
            transition.setAlpha(view: innerBackgroundView, alpha: 0.0, completion: { [weak innerBackgroundView] _ in
                innerBackgroundView?.removeFromSuperview()
            })
            transition.setScale(view: innerBackgroundView, scale: 0.001)
            
            innerBackgroundView.removeFromSuperview()
        }
        
        let params = Params(shape: shape, isDark: isDark, tintColor: tintColor, isInteractive: isInteractive)
        if self.params != params {
            self.params = params
            
            let outerCornerRadius: CGFloat
            switch shape {
            case let .roundedRect(cornerRadius):
                outerCornerRadius = cornerRadius
            }
            
            if let shadowView = self.shadowView {
                let shadowInnerInset: CGFloat = 0.5
                shadowView.image = generateImage(CGSize(width: shadowInset * 2.0 + outerCornerRadius * 2.0, height: shadowInset * 2.0 + outerCornerRadius * 2.0), rotatedContext: { size, context in
                    context.clear(CGRect(origin: CGPoint(), size: size))
                    
                    context.setFillColor(UIColor.black.cgColor)
                    context.setShadow(offset: CGSize(width: 0.0, height: 1.0), blur: 40.0, color: UIColor(white: 0.0, alpha: 0.04).cgColor)
                    context.fillEllipse(in: CGRect(origin: CGPoint(x: shadowInset + shadowInnerInset, y: shadowInset + shadowInnerInset), size: CGSize(width: size.width - shadowInset * 2.0 - shadowInnerInset * 2.0, height: size.height - shadowInset * 2.0 - shadowInnerInset * 2.0)))
                    
                    context.setFillColor(UIColor.clear.cgColor)
                    context.setBlendMode(.copy)
                    context.fillEllipse(in: CGRect(origin: CGPoint(x: shadowInset + shadowInnerInset, y: shadowInset + shadowInnerInset), size: CGSize(width: size.width - shadowInset * 2.0 - shadowInnerInset * 2.0, height: size.height - shadowInset * 2.0 - shadowInnerInset * 2.0)))
                })?.stretchableImage(withLeftCapWidth: Int(shadowInset + outerCornerRadius), topCapHeight: Int(shadowInset + outerCornerRadius))
            }
            
            if let foregroundView = self.foregroundView {
                foregroundView.image = GlassBackgroundView.generateLegacyGlassImage(size: CGSize(width: outerCornerRadius * 2.0, height: outerCornerRadius * 2.0), inset: shadowInset, isDark: isDark, fillColor: tintColor.color)
            } else {
                if let nativeParamsView = self.nativeParamsView, let nativeView = self.nativeView {
                    if #available(iOS 26.0, *) {
                        let glassEffect = UIGlassEffect(style: .regular)
                        switch tintColor.kind {
                        case .panel:
                            glassEffect.tintColor = UIColor(white: isDark ? 0.0 : 1.0, alpha: 0.1)
                        case .custom:
                            glassEffect.tintColor = tintColor.color
                        }
                        glassEffect.isInteractive = params.isInteractive
                        
                        if transition.animation.isImmediate {
                            nativeView.effect = glassEffect
                        } else {
                            UIView.animate(withDuration: 0.2, animations: {
                                nativeView.effect = glassEffect
                            })
                        }
                        
                        if isDark {
                            nativeParamsView.lumaMin = 0.0
                            nativeParamsView.lumaMax = 0.15
                        } else {
                            nativeParamsView.lumaMin = 0.6
                            nativeParamsView.lumaMax = 0.61
                        }
                    }
                }
            }
        }
        
        transition.setFrame(view: self.maskContainerView, frame: CGRect(origin: CGPoint(), size: CGSize(width: size.width + shadowInset * 2.0, height: size.height + shadowInset * 2.0)))
        transition.setFrame(view: self.maskContentView, frame: CGRect(origin: CGPoint(x: shadowInset, y: shadowInset), size: size))
        if let foregroundView = self.foregroundView {
            transition.setFrame(view: foregroundView, frame: CGRect(origin: CGPoint(), size: size).insetBy(dx: -shadowInset, dy: -shadowInset))
        }
        if let shadowView = self.shadowView {
            transition.setFrame(view: shadowView, frame: CGRect(origin: CGPoint(), size: size).insetBy(dx: -shadowInset, dy: -shadowInset))
        }
        transition.setFrame(view: self.contentContainer, frame: CGRect(origin: CGPoint(), size: size))
    }
}

public final class GlassBackgroundContainerView: UIView {
    private final class ContentView: UIView {
    }
    
    private let legacyView: ContentView?
    private let nativeParamsView: EffectSettingsContainerView?
    private let nativeView: UIVisualEffectView?
    
    public var contentView: UIView {
        if let nativeView = self.nativeView {
            return nativeView.contentView
        } else {
            return self.legacyView!
        }
    }
    
    public override init(frame: CGRect) {
        if #available(iOS 26.0, *) {
            let effect = UIGlassContainerEffect()
            effect.spacing = 7.0
            let nativeView = UIVisualEffectView(effect: effect)
            self.nativeView = nativeView
            
            let nativeParamsView = EffectSettingsContainerView(frame: CGRect())
            self.nativeParamsView = nativeParamsView
            nativeParamsView.addSubview(nativeView)
            
            self.legacyView = nil
        } else {
            self.nativeView = nil
            self.nativeParamsView = nil
            self.legacyView = ContentView()
        }
        
        super.init(frame: frame)
        
        if let nativeParamsView = self.nativeParamsView {
            self.addSubview(nativeParamsView)
        } else if let legacyView = self.legacyView {
            self.addSubview(legacyView)
        }
    }
    
    required public init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override public func didAddSubview(_ subview: UIView) {
        super.didAddSubview(subview)
        
        if subview !== self.nativeParamsView && subview !== self.legacyView {
            assertionFailure()
        }
    }
    
    override public func hitTest(_ point: CGPoint, with event: UIEvent?) -> UIView? {
        guard let result = self.contentView.hitTest(point, with: event) else {
            return nil
        }
        return result
    }
    
    public func update(size: CGSize, isDark: Bool, transition: ComponentTransition) {
        if let nativeParamsView = self.nativeParamsView, let nativeView = self.nativeView {
            nativeView.overrideUserInterfaceStyle = isDark ? .dark : .light
            
            if isDark {
                nativeParamsView.lumaMin = 0.0
                nativeParamsView.lumaMax = 0.15
            } else {
                nativeParamsView.lumaMin = 0.6
                nativeParamsView.lumaMax = 0.61
            }
            
            transition.setFrame(view: nativeView, frame: CGRect(origin: CGPoint(), size: size))
        } else if let legacyView = self.legacyView {
            transition.setFrame(view: legacyView, frame: CGRect(origin: CGPoint(), size: size))
        }
    }
}

private extension CGContext {
    func addBadgePath(in rect: CGRect) {
        saveGState()
        translateBy(x: rect.minX, y: rect.minY)
        scaleBy(x: rect.width / 78.0, y: rect.height / 78.0)
        
        // M 0 39
        move(to: CGPoint(x: 0, y: 39))
        
        // C 0 17.4609 17.4609 0 39 0
        addCurve(to: CGPoint(x: 39, y: 0),
                 control1: CGPoint(x: 0,       y: 17.4609),
                 control2: CGPoint(x: 17.4609, y: 0))
        
        // H 42
        addLine(to: CGPoint(x: 42, y: 0))
        
        // C 61.8823 0 78 16.1177 78 36
        addCurve(to: CGPoint(x: 78, y: 36),
                 control1: CGPoint(x: 61.8823, y: 0),
                 control2: CGPoint(x: 78,      y: 16.1177))
        
        // V 39
        addLine(to: CGPoint(x: 78, y: 39))
        
        // C 78 60.5391 60.5391 78 39 78
        addCurve(to: CGPoint(x: 39, y: 78),
                 control1: CGPoint(x: 78,      y: 60.5391),
                 control2: CGPoint(x: 60.5391, y: 78))
        
        // H 36
        addLine(to: CGPoint(x: 36, y: 78))
        
        // C 16.1177 78 0 61.8823 0 42
        addCurve(to: CGPoint(x: 0, y: 42),
                 control1: CGPoint(x: 16.1177, y: 78),
                 control2: CGPoint(x: 0,       y: 61.8823))
        
        // V 39 / Z
        addLine(to: CGPoint(x: 0, y: 39))
        closePath()
        
        restoreGState()
    }
}

public extension GlassBackgroundView {
    static func generateLegacyGlassImage(size: CGSize, inset: CGFloat, isDark: Bool, fillColor: UIColor) -> UIImage {
        var size = size
        if size == .zero {
            size = CGSize(width: 2.0, height: 2.0)
        }
        let innerSize = size
        size.width += inset * 2.0
        size.height += inset * 2.0
        
        return UIGraphicsImageRenderer(size: size).image { ctx in
            let context = ctx.cgContext
            
            context.clear(CGRect(origin: CGPoint(), size: size))

            let addShadow: (CGContext, Bool, CGPoint, CGFloat, CGFloat, UIColor, CGBlendMode) -> Void = { context, isOuter, position, blur, spread, shadowColor, blendMode in
                var blur = blur
                
                if isOuter {
                    blur += abs(spread)
                    
                    context.beginTransparencyLayer(auxiliaryInfo: nil)
                    context.saveGState()
                    defer {
                        context.restoreGState()
                        context.endTransparencyLayer()
                    }

                    let spreadRect = CGRect(origin: CGPoint(x: inset, y: inset), size: innerSize).insetBy(dx: 0.25, dy: 0.25)
                    let spreadPath = UIBezierPath(
                        roundedRect: spreadRect,
                        cornerRadius: min(spreadRect.width, spreadRect.height) * 0.5
                    ).cgPath

                    context.setShadow(offset: CGSize(width: position.x, height: position.y), blur: blur, color: shadowColor.cgColor)
                    context.setFillColor(UIColor.black.withAlphaComponent(1.0).cgColor)
                    context.addPath(spreadPath)
                    context.fillPath()
                    
                    let cleanRect = CGRect(origin: CGPoint(x: inset, y: inset), size: innerSize)
                    let cleanPath = UIBezierPath(
                        roundedRect: cleanRect,
                        cornerRadius: min(cleanRect.width, cleanRect.height) * 0.5
                    ).cgPath
                    context.setBlendMode(.copy)
                    context.setFillColor(UIColor.clear.cgColor)
                    context.addPath(cleanPath)
                    context.fillPath()
                    context.setBlendMode(.normal)
                } else {
                    let image = UIGraphicsImageRenderer(size: size).image(actions: { ctx in
                        let context = ctx.cgContext
                        
                        context.clear(CGRect(origin: CGPoint(), size: size))
                        let spreadRect = CGRect(origin: CGPoint(x: inset, y: inset), size: innerSize).insetBy(dx: -spread - 0.33, dy: -spread - 0.33)

                        context.setShadow(offset: CGSize(width: position.x, height: position.y), blur: blur, color: shadowColor.cgColor)
                        context.setFillColor(shadowColor.cgColor)
                        let enclosingRect = spreadRect.insetBy(dx: -10000.0, dy: -10000.0)
                        context.addPath(UIBezierPath(rect: enclosingRect).cgPath)
                        context.addBadgePath(in: spreadRect)
                        context.fillPath(using: .evenOdd)
                    })
                    
                    UIGraphicsPushContext(context)
                    image.draw(in: CGRect(origin: .zero, size: size), blendMode: blendMode, alpha: 1.0)
                    UIGraphicsPopContext()
                }
            }
            
            addShadow(context, true, CGPoint(), 10.0, 0.0, UIColor(white: 0.0, alpha: 0.06), .normal)
            addShadow(context, true, CGPoint(), 20.0, 0.0, UIColor(white: 0.0, alpha: 0.06), .normal)
            
            var a: CGFloat = 0.0
            var b: CGFloat = 0.0
            var s: CGFloat = 0.0
            fillColor.getHue(nil, saturation: &s, brightness: &b, alpha: &a)
            
            let innerImage: UIImage
            if size == CGSize(width: 40.0 + inset * 2.0, height: 40.0 + inset * 2.0), b >= 0.2 {
                innerImage = UIGraphicsImageRenderer(size: size).image { ctx in
                    let context = ctx.cgContext
                    
                    context.setFillColor(fillColor.cgColor)
                    context.fill(CGRect(origin: CGPoint(), size: size))
                    
                    if let image = UIImage(bundleImageName: "Item List/GlassEdge40x40") {
                        let imageInset = (image.size.width - 40.0) * 0.5
                        
                        if s == 0.0 && abs(a - 0.7) < 0.1 && !isDark {
                            image.draw(in: CGRect(origin: CGPoint(), size: size).insetBy(dx: inset - imageInset, dy: inset - imageInset), blendMode: .normal, alpha: 1.0)
                        } else if s <= 0.3 && !isDark {
                            image.draw(in: CGRect(origin: CGPoint(), size: size).insetBy(dx: inset - imageInset, dy: inset - imageInset), blendMode: .normal, alpha: 0.7)
                        } else if b >= 0.2 {
                            let maxAlpha: CGFloat = isDark ? 0.7 : 0.8
                            image.draw(in: CGRect(origin: CGPoint(), size: size).insetBy(dx: inset - imageInset, dy: inset - imageInset), blendMode: .overlay, alpha: max(0.5, min(1.0, maxAlpha * s)))
                        } else {
                            image.draw(in: CGRect(origin: CGPoint(), size: size).insetBy(dx: inset - imageInset, dy: inset - imageInset), blendMode: .normal, alpha: 0.5)
                        }
                    }
                }
            } else {
                innerImage = UIGraphicsImageRenderer(size: size).image { ctx in
                    let context = ctx.cgContext
                    
                    context.setFillColor(fillColor.cgColor)
                    context.fill(CGRect(origin: CGPoint(), size: size).insetBy(dx: inset, dy: inset).insetBy(dx: 0.1, dy: 0.1))
                    
                    addShadow(context, true, CGPoint(x: 0.0, y: 0.0), 20.0, 0.0, UIColor(white: 0.0, alpha: 0.04), .normal)
                    addShadow(context, true, CGPoint(x: 0.0, y: 0.0), 5.0, 0.0, UIColor(white: 0.0, alpha: 0.04), .normal)
                    
                    if s <= 0.3 && !isDark {
                        addShadow(context, false, CGPoint(x: 0.0, y: 0.0), 8.0, 0.0, UIColor(white: 0.0, alpha: 0.4), .overlay)
                        
                        let edgeAlpha: CGFloat = max(0.8, min(1.0, a))
                        
                        for _ in 0 ..< 2 {
                            addShadow(context, false, CGPoint(x: -0.64, y: -0.64), 0.8, 0.0, UIColor(white: 1.0, alpha: edgeAlpha), .normal)
                            addShadow(context, false, CGPoint(x: 0.64, y: 0.64), 0.8, 0.0, UIColor(white: 1.0, alpha: edgeAlpha), .normal)
                        }
                    } else if b >= 0.2 {
                        let edgeAlpha: CGFloat = max(0.2, min(isDark ? 0.5 : 0.7, a * a * a))
                        
                        addShadow(context, false, CGPoint(x: -0.64, y: -0.64), 0.5, 0.0, UIColor(white: 1.0, alpha: edgeAlpha), .plusLighter)
                        addShadow(context, false, CGPoint(x: 0.64, y: 0.64), 0.5, 0.0, UIColor(white: 1.0, alpha: edgeAlpha), .plusLighter)
                    } else {
                        let edgeAlpha: CGFloat = max(0.4, min(isDark ? 0.5 : 0.7, a * a * a))
                        
                        addShadow(context, false, CGPoint(x: -0.64, y: -0.64), 1.2, 0.0, UIColor(white: 1.0, alpha: edgeAlpha), .normal)
                        addShadow(context, false, CGPoint(x: 0.64, y: 0.64), 1.2, 0.0, UIColor(white: 1.0, alpha: edgeAlpha), .normal)
                    }
                }
            }
            
            context.addEllipse(in: CGRect(origin: CGPoint(x: inset, y: inset), size: innerSize))
            context.clip()
            innerImage.draw(in: CGRect(origin: CGPoint(), size: size))
        }.stretchableImage(withLeftCapWidth: Int(size.width * 0.5), topCapHeight: Int(size.height * 0.5))
    }
    
    static func generateForegroundImage(size: CGSize, isDark: Bool, fillColor: UIColor) -> UIImage {
        var size = size
        if size == .zero {
            size = CGSize(width: 1.0, height: 1.0)
        }
        
        return generateImage(size, rotatedContext: { size, context in
            context.clear(CGRect(origin: CGPoint(), size: size))
            
            let maxColor = UIColor(white: 1.0, alpha: isDark ? 0.2 : 0.9)
            let minColor = UIColor(white: 1.0, alpha: 0.0)
            
            context.setFillColor(fillColor.cgColor)
            context.fillEllipse(in: CGRect(origin: CGPoint(), size: size))
            
            let lineWidth: CGFloat = isDark ? 0.33 : 0.66
            
            context.saveGState()
            
            let darkShadeColor = UIColor(white: isDark ? 1.0 : 0.0, alpha: isDark ? 0.0 : 0.035)
            let lightShadeColor = UIColor(white: isDark ? 0.0 : 1.0, alpha: isDark ? 0.0 : 0.035)
            let innerShadowBlur: CGFloat = 24.0
            
            context.resetClip()
            context.addEllipse(in: CGRect(origin: CGPoint(), size: size).insetBy(dx: lineWidth * 0.5, dy: lineWidth * 0.5))
            context.clip()
            context.addRect(CGRect(origin: CGPoint(), size: size).insetBy(dx: -100.0, dy: -100.0))
            context.addEllipse(in: CGRect(origin: CGPoint(), size: size))
            context.setFillColor(UIColor.black.cgColor)
            context.setShadow(offset: CGSize(width: 10.0, height: -10.0), blur: innerShadowBlur, color: darkShadeColor.cgColor)
            context.fillPath(using: .evenOdd)
            
            context.resetClip()
            context.addEllipse(in: CGRect(origin: CGPoint(), size: size).insetBy(dx: lineWidth * 0.5, dy: lineWidth * 0.5))
            context.clip()
            context.addRect(CGRect(origin: CGPoint(), size: size).insetBy(dx: -100.0, dy: -100.0))
            context.addEllipse(in: CGRect(origin: CGPoint(), size: size))
            context.setFillColor(UIColor.black.cgColor)
            context.setShadow(offset: CGSize(width: -10.0, height: 10.0), blur: innerShadowBlur, color: lightShadeColor.cgColor)
            context.fillPath(using: .evenOdd)
            
            context.restoreGState()
            
            context.setLineWidth(lineWidth)
            
            context.addRect(CGRect(origin: CGPoint(x: 0.0, y: 0.0), size: CGSize(width: size.width * 0.5, height: size.height)))
            context.clip()
            context.addEllipse(in: CGRect(origin: CGPoint(), size: size).insetBy(dx: lineWidth * 0.5, dy: lineWidth * 0.5))
            context.replacePathWithStrokedPath()
            context.clip()
            
            do {
                var locations: [CGFloat] = [0.0, 0.5, 0.5 + 0.2, 1.0 - 0.1, 1.0]
                let colors: [CGColor] = [maxColor.cgColor, maxColor.cgColor, minColor.cgColor, minColor.cgColor, maxColor.cgColor]
                
                let colorSpace = CGColorSpaceCreateDeviceRGB()
                let gradient = CGGradient(colorsSpace: colorSpace, colors: colors as CFArray, locations: &locations)!
                
                context.drawLinearGradient(gradient, start: CGPoint(x: 0.0, y: 0.0), end: CGPoint(x: 0.0, y: size.height), options: CGGradientDrawingOptions())
            }
            
            context.resetClip()
            context.addRect(CGRect(origin: CGPoint(x: size.width - size.width * 0.5, y: 0.0), size: CGSize(width: size.width * 0.5, height: size.height)))
            context.clip()
            context.addEllipse(in: CGRect(origin: CGPoint(), size: size).insetBy(dx: lineWidth * 0.5, dy: lineWidth * 0.5))
            context.replacePathWithStrokedPath()
            context.clip()
            
            do {
                var locations: [CGFloat] = [0.0, 0.1, 0.5 - 0.2, 0.5, 1.0]
                let colors: [CGColor] = [maxColor.cgColor, minColor.cgColor, minColor.cgColor, maxColor.cgColor, maxColor.cgColor]
                
                let colorSpace = CGColorSpaceCreateDeviceRGB()
                let gradient = CGGradient(colorsSpace: colorSpace, colors: colors as CFArray, locations: &locations)!
                
                context.drawLinearGradient(gradient, start: CGPoint(x: 0.0, y: 0.0), end: CGPoint(x: 0.0, y: size.height), options: CGGradientDrawingOptions())
            }
        })!.stretchableImage(withLeftCapWidth: Int(size.width * 0.5), topCapHeight: Int(size.height * 0.5))
    }
}

public final class GlassBackgroundComponent: Component {
    private let size: CGSize
    private let cornerRadius: CGFloat
    private let isDark: Bool
    private let tintColor: GlassBackgroundView.TintColor
    
    public init(size: CGSize, cornerRadius: CGFloat, isDark: Bool, tintColor: GlassBackgroundView.TintColor) {
        self.size = size
        self.cornerRadius = cornerRadius
        self.isDark = isDark
        self.tintColor = tintColor
    }
    
    public static func == (lhs: GlassBackgroundComponent, rhs: GlassBackgroundComponent) -> Bool {
        if lhs.size != rhs.size {
            return false
        }
        if lhs.cornerRadius != rhs.cornerRadius {
            return false
        }
        if lhs.isDark != rhs.isDark {
            return false
        }
        if lhs.tintColor != rhs.tintColor {
            return false
        }
        return true
    }
    
    public final class View: GlassBackgroundView {
        func update(component: GlassBackgroundComponent, availableSize: CGSize, state: EmptyComponentState, environment: Environment<Empty>, transition: ComponentTransition) -> CGSize {
            self.update(size: component.size, cornerRadius: component.cornerRadius, isDark: component.isDark, tintColor: component.tintColor, transition: transition)
            
            return component.size
        }
    }
    
    public func makeView() -> View {
        return View()
    }
    
    public func update(view: View, availableSize: CGSize, state: EmptyComponentState, environment: Environment<EnvironmentType>, transition: ComponentTransition) -> CGSize {
        return view.update(component: self, availableSize: availableSize, state: state, environment: environment, transition: transition)
    }
}
