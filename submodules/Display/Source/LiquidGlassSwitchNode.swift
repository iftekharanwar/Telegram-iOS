import Foundation
import UIKit
import AsyncDisplayKit

open class LiquidGlassSwitchNode: ASDisplayNode {
    public var valueUpdated: ((Bool) -> Void)?

    public var frameColor = UIColor(rgb: 0xe0e0e0) {
        didSet {
            if self.isNodeLoaded {
                if let switchView = self.view as? LiquidGlassSwitchView {
                    switchView.offTintColor = self.frameColor
                }
            }
        }
    }

    public var handleColor = UIColor(rgb: 0xffffff) {
        didSet {
        }
    }

    public var contentColor = UIColor(rgb: 0x42d451) {
        didSet {
            if self.isNodeLoaded {
                if let switchView = self.view as? LiquidGlassSwitchView {
                    switchView.onTintColor = self.contentColor
                }
            }
        }
    }

    private var _isOn: Bool = false
    public var isOn: Bool {
        get {
            return self._isOn
        }
        set(value) {
            if value != self._isOn {
                self._isOn = value
                if self.isNodeLoaded {
                    (self.view as? LiquidGlassSwitchView)?.setOn(value, animated: false)
                }
            }
        }
    }

    override public init() {
        super.init()

        self.setViewBlock({
            return LiquidGlassSwitchView()
        })
    }

    override open func didLoad() {
        super.didLoad()

        self.view.isAccessibilityElement = false

        if let switchView = self.view as? LiquidGlassSwitchView {
            switchView.offTintColor = self.frameColor
            switchView.onTintColor = self.contentColor
            switchView.setOn(self._isOn, animated: false)

            switchView.valueChanged = { [weak self] value in
                self?._isOn = value
                self?.valueUpdated?(value)
            }

            switchView.addTarget(self, action: #selector(switchValueChanged(_:)), for: .valueChanged)
        }
    }

    public func setOn(_ value: Bool, animated: Bool) {
        self._isOn = value
        if self.isNodeLoaded {
            (self.view as? LiquidGlassSwitchView)?.setOn(value, animated: animated)
        }
    }

    override open func calculateSizeThatFits(_ constrainedSize: CGSize) -> CGSize {
        return CGSize(width: 51.0, height: 31.0)
    }

    @objc private func switchValueChanged(_ switchView: LiquidGlassSwitchView) {
        self._isOn = switchView.isOn
        self.valueUpdated?(switchView.isOn)
    }
}
