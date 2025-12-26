import Foundation
import UIKit
import Display
import AsyncDisplayKit
import ComponentFlow
import TelegramPresentationData

public final class SwitchComponent: Component {
    public typealias EnvironmentType = Empty

    let tintColor: UIColor?
    let value: Bool
    let useLiquidGlass: Bool
    let valueUpdated: (Bool) -> Void

    public init(
        tintColor: UIColor? = nil,
        value: Bool,
        useLiquidGlass: Bool = true,
        valueUpdated: @escaping (Bool) -> Void
    ) {
        self.tintColor = tintColor
        self.value = value
        self.useLiquidGlass = useLiquidGlass
        self.valueUpdated = valueUpdated
    }

    public static func ==(lhs: SwitchComponent, rhs: SwitchComponent) -> Bool {
        if lhs.tintColor != rhs.tintColor {
            return false
        }
        if lhs.value != rhs.value {
            return false
        }
        if lhs.useLiquidGlass != rhs.useLiquidGlass {
            return false
        }
        return true
    }

    public final class View: UIView {
        private var switchView: UISwitch?
        private var liquidGlassSwitch: LiquidGlassSwitchView?

        private var component: SwitchComponent?

        override init(frame: CGRect) {
            super.init(frame: frame)
        }

        required init?(coder: NSCoder) {
            fatalError("init(coder:) has not been implemented")
        }

        @objc func valueChanged(_ sender: Any) {
            if let switchView = self.switchView {
                self.component?.valueUpdated(switchView.isOn)
            } else if let liquidGlassSwitch = self.liquidGlassSwitch {
                self.component?.valueUpdated(liquidGlassSwitch.isOn)
            }
        }

        func update(component: SwitchComponent, availableSize: CGSize, state: EmptyComponentState, environment: Environment<EnvironmentType>, transition: ComponentTransition) -> CGSize {
            self.component = component

            if component.useLiquidGlass {
                if self.liquidGlassSwitch == nil {
                    let liquidSwitch = LiquidGlassSwitchView()
                    self.liquidGlassSwitch = liquidSwitch
                    self.addSubview(liquidSwitch)

                    liquidSwitch.valueChanged = { [weak self] value in
                        self?.component?.valueUpdated(value)
                    }

                    self.switchView?.removeFromSuperview()
                    self.switchView = nil
                }

                if let liquidSwitch = self.liquidGlassSwitch {
                    liquidSwitch.isOn = component.value
                    liquidSwitch.sizeToFit()
                    liquidSwitch.frame = CGRect(origin: .zero, size: liquidSwitch.frame.size)
                    return liquidSwitch.frame.size
                }
            } else {
                if self.switchView == nil {
                    let standardSwitch = UISwitch()
                    self.switchView = standardSwitch
                    self.addSubview(standardSwitch)
                    standardSwitch.addTarget(self, action: #selector(self.valueChanged(_:)), for: .valueChanged)

                    self.liquidGlassSwitch?.removeFromSuperview()
                    self.liquidGlassSwitch = nil
                }

                if let standardSwitch = self.switchView {
                    standardSwitch.tintColor = component.tintColor
                    standardSwitch.setOn(component.value, animated: !transition.animation.isImmediate)
                    standardSwitch.sizeToFit()
                    standardSwitch.frame = CGRect(origin: .zero, size: standardSwitch.frame.size)
                    return standardSwitch.frame.size
                }
            }

            return CGSize(width: 64, height: 31)
        }
    }

    public func makeView() -> View {
        return View(frame: CGRect())
    }

    public func update(view: View, availableSize: CGSize, state: EmptyComponentState, environment: Environment<EnvironmentType>, transition: ComponentTransition) -> CGSize {
        return view.update(component: self, availableSize: availableSize, state: state, environment: environment, transition: transition)
    }
}
